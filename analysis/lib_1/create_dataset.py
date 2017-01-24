import csv
from collections import defaultdict
import json
import urllib
from datetime import datetime
# from difflib import SequenceMatcher
import codecs
import os
SCRIPT_DIR = os.path.dirname(__file__)

# from prediction import generate_X_Y_dict

import unicodedata

# # !!! remove hardcoding !!!

player_type_dict = [
    {
        'id': 1,
        'singular_name': "Goalkeeper",
        'singular_name_short': "GKP",
        'plural_name': "Goalkeepers",
        'plural_name_short': "GKP"
    },
    {
        'id': 2,
        'singular_name': "Defender",
        'singular_name_short': "DEF",
        'plural_name': "Defenders",
        'plural_name_short': "DEF"
    },
    {
        'id': 3,
        'singular_name': "Midfielder",
        'singular_name_short': "MID",
        'plural_name': "Midfielders",
        'plural_name_short': "MID"
    },
    {
        'id': 4,
        'singular_name': "Forward",
        'singular_name_short': "FWD",
        'plural_name': "Forwards",
        'plural_name_short': "FWD"
    }
]

teams_points_per_match_2015 = {
    1: float(71) / 38,                      # Arsenal
    2: float(42) / 38,                      # Bournemouth
    3: float(37) / 38,                      # Burnley
    4: float(50) / 38,                      # Chelsea
    5: float(42) / 38,                      # Crystal Palace
    6: float(47) / 38,                      # Everton
    7: float(37) / 38,                      # Hull
    8: float(81) / 38,                      # Leicester
    9: float(60) / 38,                      # Liverpool
    10: float(66) / 38,                     # Man City
    11: float(66) / 38,                     # Man Utd
    12: float(37) / 38,                     # Middlesbrough
    13: float(63) / 38,                     # Southampton
    14: float(51) / 38,                     # Stoke
    15: float(38) / 38,                     # Sunderland
    16: float(47) / 38,                     # Swansea
    17: float(70) / 38,                     # Spurs
    18: float(45) / 38,                     # Watford
    19: float(43) / 38,                     # West Brom
    20: float(62) / 38,                     # West Ham
}

fpl_to_football_data_org_team_map = {
    1: 57,
    2: 1044,
    3: 328,
    4: 61,
    5: 354,
    6: 62,
    7: 322,
    8: 338,
    9: 64,
    10: 65,
    11: 66,
    12: 343,
    13: 340,
    14: 70,
    15: 71,
    16: 72,
    17: 73,
    18: 346,
    19: 74,
    20: 563,
}


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def update_2016_fpl_data():
    """downloads and updates FPL data for 2016-2017 season using fpl api
    """
    master_url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
    response = urllib.urlopen(master_url)
    master_data = json.loads(response.read())
    print "master data loaded"
    i = 0
    for player in master_data['elements']:
        player_id = player['id']
        print 'pulling data for player %s' % (i)
        player_url = 'https://fantasy.premierleague.com/drf/element-summary/%s' % (player_id)
        response = urllib.urlopen(player_url)
        player_data = json.loads(response.read())
        master_data['elements'][i]['player_data'] = player_data
        i += 1

    data_json = json.dumps(master_data)
    with open('fpl_2016_data.json', 'w') as f:
        f.write(data_json)


def create_table_dataset():
    """creates a file containing json of form -
    {
        "Liverpool",
        {
            "2013": [
                {
                    "week": 1,
                    "total_points": 0,
                    "total_goals_scored": 0,
                    "total_goals_conceded": 0,
                    "total_clean_sheets": 0,
                    "at_home": True,
                },
                ...
            ]
            ...
        },
        ...
    },
    """
    with open('england_2013-2015_fixtures.csv', 'r') as read_file:
        # create list of fixtures per team
        fixtures_dict = defaultdict(list)
        next(read_file)
        for line in read_file:
            entries = [entry.strip().strip('"') for entry in line.split(',')]
            date = datetime.strptime(entries[0], '%Y-%m-%d')
            season = int(entries[1])
            home_team = entries[2]
            away_team = entries[3]
            home_goals = int(entries[4])
            away_goals = int(entries[5])
            if home_goals > away_goals:
                points_tuple = (3, 0)
            elif home_goals == away_goals:
                points_tuple = (1, 1)
            else:
                points_tuple = (0, 3)
            home_entry = {
                'date': date,
                'season': season,
                'opponent': away_team,
                'is_home': True,
                'goals_scored': home_goals,
                'goals_conceded': away_goals,
                'points_taken': points_tuple[0]
            }
            away_entry = {
                'date': date,
                'season': season,
                'opponent': home_team,
                'is_home': False,
                'goals_scored': away_goals,
                'goals_conceded': home_goals,
                'points_taken': points_tuple[1]
            }
            fixtures_dict[home_team].append(home_entry)
            fixtures_dict[away_team].append(away_entry)

    # print fixtures_dict['Liverpool']
    teams_dataset_dict = {}
    for team, fixtures in fixtures_dict.iteritems():
        seasons_dict = defaultdict(list)
        sorted_fixtures = sorted(fixtures, key=lambda k: k['date'])
        i = 0
        total_points = 0
        total_clean_sheets = 0
        total_goals_scored = 0
        total_goals_conceded = 0
        previous_season = 2012
        for result in sorted_fixtures:
            season = result['season']
            if previous_season != season:
                i = 0
                total_points = 0
                total_goals_scored = 0
                total_goals_conceded = 0
                total_clean_sheets = 0
                previous_season = season
            total_points += result['points_taken']
            total_goals_scored += result['goals_scored']
            total_goals_conceded += result['goals_conceded']
            if result['goals_conceded'] == 0:
                total_clean_sheets += 1
            i += 1
            week = i
            entry = {
                "week": week,
                "total_points": total_points,
                "total_goals_scored": total_goals_scored,
                "total_goals_conceded": total_goals_conceded,
                "total_clean_sheets": total_clean_sheets,
                "at_home": result['is_home']
            }
            seasons_dict[season].append(entry)
        teams_dataset_dict[team] = seasons_dict
    teams_dataset_json = json.dumps(teams_dataset_dict)
    with open('table_data.json', 'w') as f:
        f.write(teams_dataset_json)

    # print teams_dataset_dict['Liverpool'][2014][37]


def create_players_static_dataset_2014():
    """creates a json list of dicts with following values per player for each player in fpl 2016 db
        1:{
            'name': name,
            'position': position,
            'team_name': team_name,
            'points_last_season': points_last_season (None if not in last season roster),
            'minutes_last_season': minutes_last_season (None if not in 2013 roster),
        }
    """
    # load 2014 fpl player data
    fpl14_player_list = []
    with codecs.open('FPL2014/Player_Details.csv', 'r', encoding='utf-8', errors='replace') as f:
        next(f)
        for line in f:
            entries = [entry.strip().strip('"') for entry in line.split(',')]
            fpl14_player_list.append(entries)

    # load 2013 fpl player data
    fpl13_player_list = []
    with codecs.open('FPL2013/FPL2013_player_data.csv', 'r', encoding='utf-8', errors='replace') as f:
        next(f)
        for line in f:
            entries = [entry.strip().strip('"') for entry in line.split(',')]
            fpl13_player_list.append(entries)

    player_dict = defaultdict(dict)
    print len(fpl14_player_list)
    for player_entry in fpl14_player_list:
        # print player_entry
        id = int(player_entry[0])
        name = player_entry[1]
        print name
        # name = name.decode('string_escape')
        # name = name.encode('utf8')
        # print name
        team_name = player_entry[2]
        position = player_entry[3]
        i = 0
        index = -1
        for player_13 in fpl13_player_list:
            if name.lower().strip() == player_13[0].lower().strip():
                index = i
            i += 1
        if index == -1:
            points_last_season = None
            minutes_last_season = None
        else:
            matched_player = fpl13_player_list.pop(index)
            points_last_season = matched_player[6]
            minutes_last_season = matched_player[2]
        player_dict[id] = {
            'name': name,
            'team_name': team_name,
            'position': position,
            'points_last_season': points_last_season,
            'minutes_last_season': minutes_last_season,
        }
    player_dict_json = json.dumps(player_dict)
    with open('2014_static_player_dataset.json', 'w') as f:
        f.write(player_dict_json)


def get_average_points_per_minute_dict(players_static_data):
    """returns dict with average points per last season per position
    """
    total_pls_goalkeeper = 0.0
    total_minls_goalkeeper = 0.0
    # num_of_goalkeepers = 0
    total_pls_defender = 0.0
    total_minls_defender = 0.0
    # num_of_defenders = 0
    total_pls_midfielder = 0.0
    total_minls_midfielder = 0.0
    # num_of_midfielders = 0
    total_pls_forward = 0.0
    total_minls_forward = 0.0
    # num_of_forwards = 0

    for p_id, stats in players_static_data.iteritems():
        if stats['points_last_season']:
            if stats['position'] == 'Goalkeeper':
                # num_of_goalkeepers += 1
                total_pls_goalkeeper += float(stats['points_last_season'])
                total_minls_goalkeeper += float(stats['minutes_last_season'])
            elif stats['position'] == 'Defender':
                # num_of_defenders += 1
                total_pls_defender += float(stats['points_last_season'])
                total_minls_defender += float(stats['minutes_last_season'])
            elif stats['position'] == 'Midfielder':
                # num_of_midfielders += 1
                total_pls_midfielder += float(stats['points_last_season'])
                total_minls_midfielder += float(stats['minutes_last_season'])
            elif stats['position'] == 'Forward':
                # num_of_forwards += 1
                total_pls_forward += float(stats['points_last_season'])
                total_minls_forward += float(stats['minutes_last_season'])
            else:
                print "Something is wrong. Check player static data positions."

    average_ppmls_goalkeeper = total_pls_goalkeeper / total_minls_goalkeeper
    average_ppmls_defender = total_pls_defender / total_minls_defender
    average_ppmls_midfielder = total_pls_midfielder / total_minls_midfielder
    average_ppmls_forward = total_pls_forward / total_minls_forward

    average_ppmls_dict = {
        'Goalkeeper': average_ppmls_goalkeeper,
        'Defender': average_ppmls_defender,
        'Midfielder': average_ppmls_midfielder,
        'Forward': average_ppmls_forward,
    }
    return average_ppmls_dict


def generate_X_Y_dict(player_id, player_data=None, table_data=None, fpl_master_data=None, fixture_number=0):
    """generates a dictionary with features for a player
    """
    if not player_data:
        player_url = 'https://fantasy.premierleague.com/drf/element-summary/%s' % (player_id)
        response = urllib.urlopen(player_url)
        player_data = json.loads(response.read())

    if not fpl_master_data:
        master_url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
        response = urllib.urlopen(master_url)
        fpl_master_data = json.loads(response.read())['elements']
        # fpl_master_data = items(response, 'elements.item')

    player_X_dict = {}

    player_gen_data = None
    for element in fpl_master_data:
        # print element['id']
        if int(element['id']) == player_id:
            player_gen_data = element
            break
    if not player_gen_data:
        print "no player gen data"
        return ["Sorry. Something went wrong. Try again later"]

    player_type = int(player_gen_data['element_type'])
    # player_type_dict = fpl_master_data['element_types']

    position = [entry['singular_name'] for entry in player_type_dict if int(entry['id']) == player_type][0]

    latest_fixture = player_data['fixtures'][0]
    latest_fixture_is_home = latest_fixture['is_home']
    history = player_data['history']
    print(player_gen_data['web_name'])
    # print(len(history))
    history = sorted(history, key=lambda k: k['round'])
    # print(fixture_number)
    print("history length = %s" % (len(history)))
    print("fixture number = %s" % fixture_number)
    first_round = history[0]['round']
    last_round = history[-1]['round']
    print("first round = %s" % first_round)
    print("last round = %s" % last_round)
    if fixture_number < first_round:
        print("player not entered team for this fixture")
        return "player not entered team for this fixture"
    if last_round < fixture_number:
        print("player removed from team before this fixture")
        return "player removed from team before this fixture"
    history_entry = history[fixture_number - first_round]

    is_home = history_entry['was_home']
    week = int(history_entry['round'])

    team_id = int(latest_fixture['team_h']) if latest_fixture_is_home else latest_fixture['team_a']
    opp_team_id = int(history_entry['opponent_team'])
    player_X_dict['team_points_per_match_last_season'] = teams_points_per_match_2015[team_id]
    player_X_dict['opponent_points_per_match_last_season'] = teams_points_per_match_2015[opp_team_id]

    f_path = os.path.join(SCRIPT_DIR, '2014_static_player_dataset.json')
    with open(f_path, 'r') as f:
        players_static_data = json.load(f)

    average_ppmls_dict = get_average_points_per_minute_dict(players_static_data)

    past_seasons_data = player_data['history_past']
    if '2015/16' in past_seasons_data:
        last_season_points = float(past_seasons_data['2015/16']['total_points'])
        last_season_minutes = int(past_seasons_data['2015/16']['total_minutes'])
        if last_season_minutes == 0:
            # player present but did not play last season. Assign 0
            player_X_dict['last_season_points_per_minutes'] = 0.0
        else:
            player_X_dict['last_season_points_per_minutes'] = last_season_points / last_season_points
    else:
        # not in roster last year, assign average
        player_X_dict['last_season_points_per_minutes'] = average_ppmls_dict[position]

    player_X_dict['is_at_home'] = 1 if is_home else 0
    player_X_dict['price'] = float(history_entry['value']) / 10

    # history = player_data['history']
    if len(history) < 3:
        print "not enough matches played in season, cannot predict"
        return ["not enough matches played in season, cannot predict"]
    
    history = history[:week - first_round]
    if len(history) < 3:
        print "not enough matches played, cannot predict"
        return ["not enough matches played, cannot predict"]

    # last_match_data = history[-1]

    goals_scored = 0.0
    goals_conceded = 0.0
    assists = 0.0
    bps = 0.0
    clean_sheets = 0.0
    yellow_cards = 0.0
    red_cards = 0.0
    minutes = 0.0
    saves = 0.0
    points = 0.0
    net_transfers = 0.0
    matches_played = 0
    # price = 0

    for fixture in history:
        goals_scored += float(fixture['goals_scored'])
        # print goals_scored
        goals_conceded += float(fixture['goals_conceded'])
        assists += float(fixture['assists'])
        bps += float(fixture['bps'])
        clean_sheets += float(fixture['clean_sheets'])
        yellow_cards += float(fixture['yellow_cards'])
        red_cards += float(fixture['red_cards'])
        minutes += float(fixture['minutes'])
        saves += float(fixture['saves'])
        points += float(fixture['total_points'])
        net_transfers += float(fixture['transfers_balance'])
        matches_played += 1 if float(fixture['minutes']) > 0 else 0

    if matches_played < 3:
        print "not enough matches played for prediction"
        return ["not enough matches played for prediction"]
    player_X_dict['goals_scored_per_match_played'] = goals_scored / matches_played
    player_X_dict['goals_conceded_per_match_played'] = goals_conceded / matches_played
    player_X_dict['assists_per_match_played'] = assists / matches_played
    player_X_dict['bps_per_match_played'] = bps / matches_played
    player_X_dict['clean_sheets_per_match_played'] = clean_sheets / matches_played
    player_X_dict['yellow_cards_per_match_played'] = yellow_cards / matches_played
    player_X_dict['red_cards_per_match_played'] = red_cards / matches_played
    player_X_dict['minutes_per_match_played'] = minutes / matches_played
    player_X_dict['saves_per_match_played'] = saves / matches_played
    player_X_dict['points_per_match_played'] = points / matches_played
    player_X_dict['net_transfers_per_match_played'] = net_transfers / matches_played

    history_form = history[-3:]

    goals_scored = 0.0
    goals_conceded = 0.0
    assists = 0.0
    bps = 0.0
    clean_sheets = 0.0
    yellow_cards = 0.0
    red_cards = 0.0
    minutes = 0.0
    saves = 0.0
    points = 0.0
    net_transfers = 0.0
    matches_played = 0
    # price = 0
    for fixture in history_form:
        goals_scored += float(fixture['goals_scored'])
        goals_conceded += float(fixture['goals_conceded'])
        assists += float(fixture['assists'])
        bps += float(fixture['bps'])
        clean_sheets += float(fixture['clean_sheets'])
        yellow_cards += float(fixture['yellow_cards'])
        red_cards += float(fixture['red_cards'])
        minutes += float(fixture['minutes'])
        saves += float(fixture['saves'])
        points += float(fixture['total_points'])
        net_transfers += float(fixture['transfers_balance'])
        matches_played += 1 if float(fixture['minutes']) > 0 else 0

    if matches_played < 1:
        player_X_dict['avg_goals_scored_form'] = player_X_dict['goals_scored_per_match_played']
        player_X_dict['avg_goals_conceded_form'] = player_X_dict['goals_conceded_per_match_played']
        player_X_dict['avg_assists_form'] = player_X_dict['assists_per_match_played']
        player_X_dict['avg_bps_form'] = player_X_dict['bps_per_match_played']
        player_X_dict['avg_clean_sheets_form'] = player_X_dict['clean_sheets_per_match_played']
        player_X_dict['avg_yellow_cards_form'] = player_X_dict['yellow_cards_per_match_played']
        player_X_dict['avg_red_cards_form'] = player_X_dict['red_cards_per_match_played']
        player_X_dict['avg_minutes_form'] = player_X_dict['minutes_per_match_played']
        player_X_dict['avg_saves_form'] = player_X_dict['saves_per_match_played']
        player_X_dict['avg_points_form'] = player_X_dict['points_per_match_played']
        player_X_dict['avg_net_transfers_form'] = net_transfers / 3
    else:
        player_X_dict['avg_goals_scored_form'] = goals_scored / matches_played
        player_X_dict['avg_goals_conceded_form'] = goals_conceded / matches_played
        player_X_dict['avg_assists_form'] = assists / matches_played
        player_X_dict['avg_bps_form'] = bps / matches_played
        player_X_dict['avg_clean_sheets_form'] = clean_sheets / matches_played
        player_X_dict['avg_yellow_cards_form'] = yellow_cards / matches_played
        player_X_dict['avg_red_cards_form'] = red_cards / matches_played
        player_X_dict['avg_minutes_form'] = minutes / matches_played
        player_X_dict['avg_saves_form'] = saves / matches_played
        player_X_dict['avg_points_form'] = points / matches_played
        player_X_dict['avg_net_transfers_form'] = net_transfers / matches_played

    player_X_dict['price_change_form'] = (
        float(history_form[-1]['value']) - float(history_form[0]['value'])
    ) / len(history_form)

    if not table_data:
        matchday = week - 1
        try:
            league_table_url = 'http://api.football-data.org/v1/competitions/426/leagueTable/?matchday=%s' % matchday
        except:
            print('Sorry. Something went wrong. Try again later')
            return ["Sorry. Something went wrong. Try again later"]
        response = urllib.urlopen(league_table_url)
        table_data = json.loads(response.read())

    standings = table_data['standing']
    team_link = 'http://api.football-data.org/v1/teams/%s' % fpl_to_football_data_org_team_map[team_id]
    # print "team link = %s" % team_link
    opp_team_link = 'http://api.football-data.org/v1/teams/%s' % fpl_to_football_data_org_team_map[opp_team_id]
    team_table_data = None
    opp_team_table_data = None
    for entry in standings:
        # print entry['_links']['team']['href']
        if str(entry['_links']['team']['href']) == team_link:
            # print team_link
            team_table_data = entry
        if str(entry['_links']['team']['href']) == opp_team_link:
            # print opp_team_link
            opp_team_table_data = entry

    if not team_table_data or not opp_team_table_data:
        if not team_table_data:
            print "no team table data"
        if not opp_team_table_data:
            print "on opposition team data"
        return ["Sorry. Something went wrong. Try again later"]

    team_matches_played = int(team_table_data['playedGames'])
    opp_team_matches_played = int(opp_team_table_data['playedGames'])

    player_X_dict['team_goals_scored_per_match'] = float(team_table_data['goals']) / team_matches_played
    player_X_dict['team_goals_conceded_per_match'] = float(team_table_data['goalsAgainst']) / team_matches_played
    player_X_dict['team_points_per_match'] = float(team_table_data['points']) / team_matches_played

    player_X_dict['opponent_goals_scored_per_match'] = float(opp_team_table_data['goals']) / opp_team_matches_played
    player_X_dict['opponent_goals_conceded_per_match'] = float(opp_team_table_data['goalsAgainst']) / opp_team_matches_played
    player_X_dict['opponent_points_per_match'] = float(opp_team_table_data['points']) / opp_team_matches_played

    points_scored = int(history_entry['total_points'])

    print('data returned')
    return player_X_dict, position, points_scored


def create_data_points():
    """creates data points of the form
        player_id,
        last_season_points_per_minutes,
        opponent_points_per_game_last_season,
        minutes_per_matches_played,
        points_per_matches_played,
        goals_per_matches_played,
        assists_per_matches_played,
        yellow_cards_per_matches_played,
        red_cards_per_matches_played,
        bps_per_matches_played,
        price,
        avg_points,
        avg_minutes,
        avg_goals,
        avg_assists,
        avg_bps,
        avg_net_transfers,
        avg_price_drop,
        team_points_per_match,
        team_goals_scored_per_match,
        team_goals_conceded_per_match,
        opponent_points_per_match,
        opponent_goals_scored_per_match,
        opponent_goals_conceded_per_match,
        is_at_home,

    """
    teams_points_per_match_2013 = {
        'Arsenal': float(79) / 38,
        'Aston Villa': float(38) / 38,
        'Burnley': float(33) / 38,
        'Chelsea': float(82) / 38,
        'Everton': float(72) / 38,
        'Crystal Palace': float(45) / 38,
        'Hull': float(37) / 38,
        'Leicester': float(33) / 38,
        'Liverpool': float(84) / 38,
        'Man City': float(86) / 38,
        'Man Utd': float(64) / 38,
        'Newcastle': float(49) / 38,
        'QPR': float(33) / 38,
        'Southampton': float(56) / 38,
        'Spurs': float(69) / 38,
        'Stoke': float(50) / 38,
        'Sunderland': float(38) / 38,
        'Swansea': float(42) / 38,
        'West Brom': float(36) / 38,
        'West Ham': float(40) / 38,
    }

    f_path = os.path.join(SCRIPT_DIR, '2014_static_player_dataset.json')
    with open(f_path, 'r') as f:
        players_static_data = json.load(f)

    average_ppmls_dict = get_average_points_per_minute_dict(players_static_data)

    f_path = os.path.join(SCRIPT_DIR, 'table_data.json')
    with open(f_path, 'r') as f:
        table_data = json.load(f)

    match_data_dict = defaultdict(dict)
    f_path = os.path.join(SCRIPT_DIR, 'FPL2014/Player_Data.csv')
    with open(f_path, 'r') as f:
        next(f)
        for line in f:
            entries = [entry.strip().strip('"') for entry in line.split(',')]
            # week
            week = int(entries[1])
            # player id
            player_id = entries[0]
            # points per minutes last season
            minutes_last_season = players_static_data[player_id]['minutes_last_season']
            points_last_season = players_static_data[player_id]['points_last_season']
            if points_last_season:
                minutes_last_season = int(minutes_last_season)
                points_last_season = int(points_last_season)
                if minutes_last_season != 0:
                    last_season_points_per_minutes = float(points_last_season) / minutes_last_season
                else:
                    # player present but did not play last season
                    last_season_points_per_minutes = 0
            else:
                # new player this season, assign average values for last season
                position = players_static_data[player_id]['position']
                last_season_points_per_minutes = average_ppmls_dict[position]
            # team_points_per_match_last_season
            team = players_static_data[player_id]['team_name']
            team_points_per_match_last_season = teams_points_per_match_2013[team]
            # opponent_points_per_game_last_season
            opponent_name = entries[2]
            opponent_points_per_match_last_season = teams_points_per_match_2013[opponent_name]
            # price
            price = entries[-2]
            # team_points_per_match
            # team_goals_scored_per_match,
            # team_goals_conceded_per_match,
            # is_at_home,
            # print players_static_data[player_id]
            team_table_data = table_data[team]["2014"]
            aggregate_data = team_table_data[week - 1]
            team_points_per_match = float(aggregate_data['total_points']) / (week)
            team_goals_scored_per_match = float(aggregate_data['total_goals_scored']) / (week)
            team_goals_conceded_per_match = float(aggregate_data['total_goals_conceded']) / (week)
            # if player_id == '344':
            #     print aggregate_data['total_points']
            #     print team_points_per_match
            # print team
            # print week
            # print team_table_data[week]
            is_at_home = team_table_data[week - 1]['at_home']
            # opponent_points_per_match,
            # opponent_goals_scored_per_match,
            # opponent_goals_conceded_per_match,
            oppn_table_data = table_data[opponent_name]["2014"]
            aggregate_data = oppn_table_data[week - 1]
            opponent_points_per_match = float(aggregate_data['total_points']) / (week)
            opponent_goals_scored_per_match = float(aggregate_data['total_goals_scored']) / (week)
            opponent_goals_conceded_per_match = float(aggregate_data['total_goals_conceded']) / (week)

            player_dict = {
                'last_season_points_per_minutes': last_season_points_per_minutes,
                'team_points_per_match_last_season': team_points_per_match_last_season,
                'opponent_points_per_match_last_season': opponent_points_per_match_last_season,
                'price': price,
                'team_points_per_match': team_points_per_match,
                'team_goals_scored_per_match': team_goals_scored_per_match,
                'team_goals_conceded_per_match': team_goals_conceded_per_match,
                'is_at_home': is_at_home,
                'opponent_points_per_match': opponent_points_per_match,
                'opponent_goals_scored_per_match': opponent_goals_scored_per_match,
                'opponent_goals_conceded_per_match': opponent_goals_conceded_per_match,
                'minutes_played': int(entries[4]),
                'goals_scored': int(entries[5]),
                'assists': int(entries[6]),
                'clean_sheets': int(entries[7]),
                'goals_conceded': int(entries[8]),
                'yellow_cards': int(entries[9]),
                'red_cards': int(entries[10]),
                'saves': int(entries[11]),
                'bps': int(entries[14]),
                'net_transfers': int(entries[15]),
                'points': int(entries[17]),
            }
            match_data_dict[player_id][week] = player_dict
        # print match_data_dict['20']

    # clean player_data
    # in case of missing gameweeks, fill with previous gameweek data
    # if previous is missing, fill with next gameweek data
    deleted_player_list = []
    for key, value in match_data_dict.iteritems():
        if len(value) < 35:
            # print len(value)
            deleted_player_list.append(key)
        else:
            for i in range(1, 39, 1):
                if i not in value:
                    # print str(key)
                    # print str(i)
                    try:
                        match_data_dict[key][i] = match_data_dict[key][i - 1]
                    except KeyError:
                        if i + 1 in value:
                            match_data_dict[key][i] = match_data_dict[key][i + 1]
                        else:
                            index = min(value, key=value.get)
                            match_data_dict[key][i] = match_data_dict[key][index]

    for player_id in deleted_player_list:
        del match_data_dict[player_id]
    # for derived features
    data_dict = defaultdict(dict)
    for key, value in match_data_dict.iteritems():
        total_minutes = 0.0
        total_points = 0.0
        total_goals_scored = 0.0
        total_assists = 0.0
        total_goals_conceded = 0.0
        total_clean_sheets = 0.0
        total_yellow_cards = 0.0
        total_red_cards = 0.0
        total_saves = 0.0
        total_bps = 0.0
        total_net_transfers = 0.0
        total_matches_played = 0.0

        data_dict_per_week = defaultdict(dict)
        data_dict[key] = {
            'meta': {
                'position': players_static_data[key]['position'].lower(),
                'team': players_static_data[key]['team_name'],
                'name': players_static_data[key]['name']
            },
            'data': data_dict_per_week,
        }

        cumulative_stat_dict = {}
        for i in range(1, 39, 1):
            if i in value:
                week_stats = value[i]
                total_minutes += week_stats['minutes_played']
                total_points += week_stats['points']
                total_goals_scored += week_stats['goals_scored']
                total_assists += week_stats['assists']
                total_goals_conceded += week_stats['goals_conceded']
                total_clean_sheets += week_stats['clean_sheets']
                total_yellow_cards += week_stats['yellow_cards']
                total_red_cards += week_stats['red_cards']
                total_saves += week_stats['saves']
                total_bps += week_stats['bps']
                total_net_transfers += week_stats['net_transfers']
                total_matches_played += 1 if week_stats['minutes_played'] > 0 else 0
            else:
                pass
            # minutes_per_match_played,
            # points_per_match_played,
            # goals_scored_per_match_played,
            # assists_per_match_played,
            # yellow_cards_per_match_played,
            # red_cards_per_match_played,
            # bps_per_match_played,
            # avg_points_form
            # avg_minutes_form,
            # avg_goals_form,
            # avg_assists_form,
            # avg_bps_form,
            # avg_net_transfers_form,
            # avg_price_drop_form,
            dict_entry = {
                'total_minutes': total_minutes,
                'total_points': total_points,
                'total_goals_scored': total_goals_scored,
                'total_assists': total_assists,
                'total_goals_conceded': total_goals_conceded,
                'total_clean_sheets': total_clean_sheets,
                'total_yellow_cards': total_yellow_cards,
                'total_red_cards': total_red_cards,
                'total_saves': total_saves,
                'total_bps': total_bps,
                'total_net_transfers': total_net_transfers,
                'total_matches_played': total_matches_played,
            }
            cumulative_stat_dict[i] = dict_entry
        for i in range(4, 39, 1):
            dict_entry = {
                'minutes_per_match_played': None,
                'points_per_match_played': None,
                'goals_scored_per_match_played': None,
                'assists_per_match_played': None,
                'goals_conceded_per_match_played': None,
                'clean_sheets_per_match_played': None,
                'yellow_cards_per_match_played': None,
                'red_cards_per_match_played': None,
                'saves_per_match_played': None,
                'bps_per_match_played': None,
                'net_transfers_per_match_played': None,
                'avg_points_form': None,
                'avg_minutes_form': None,
                'avg_goals_form': None,
                'avg_assists_form': None,
                'avg_bps_form': None,
                'avg_net_transfers_form': None,
                'avg_price_drop_form': None,
            }
            cumulative_stats = cumulative_stat_dict[i - 1]
            total_matches_played = cumulative_stats['total_matches_played']
            if total_matches_played == 0:
                # not played a single match yet, discard
                next
            else:
                dict_entry = {
                    'minutes_per_match_played': cumulative_stats['total_minutes'] / total_matches_played,
                    'points_per_match_played': cumulative_stats['total_points'] / total_matches_played,
                    'goals_scored_per_match_played': cumulative_stats['total_goals_scored'] / total_matches_played,
                    'assists_per_match_played': cumulative_stats['total_assists'] / total_matches_played,
                    'goals_conceded_per_match_played': cumulative_stats['total_goals_conceded'] / total_matches_played,
                    'clean_sheets_per_match_played': cumulative_stats['total_clean_sheets'] / total_matches_played,
                    'yellow_cards_per_match_played': cumulative_stats['total_yellow_cards'] / total_matches_played,
                    'red_cards_per_match_played': cumulative_stats['total_red_cards'] / total_matches_played,
                    'saves_per_match_played': cumulative_stats['total_saves'] / total_matches_played,
                    'bps_per_match_played': cumulative_stats['total_bps'] / total_matches_played,
                    'net_transfers_per_match_played': cumulative_stats['total_net_transfers'] / total_matches_played,
                }
                # total_matches_played_3_weeks_ago = 0 if i == 4 else cumulative_stat_dict[i - 4]
                cumulative_stat_dict_3_weeks_ago = {
                    'total_minutes': 0,
                    'total_points': 0,
                    'total_goals_scored': 0,
                    'total_assists': 0,
                    'total_goals_conceded': 0,
                    'total_clean_sheets': 0,
                    'total_yellow_cards': 0,
                    'total_red_cards': 0,
                    'total_saves': 0,
                    'total_bps': 0,
                    'total_net_transfers': 0,
                    'total_matches_played': 0,
                }
                if i > 4:
                    cumulative_stat_dict_3_weeks_ago = cumulative_stat_dict[i - 4]
                total_matches_played_3_weeks_ago = cumulative_stat_dict_3_weeks_ago['total_matches_played']
                matches_played_in_last_3_weeks = total_matches_played - total_matches_played_3_weeks_ago
                if matches_played_in_last_3_weeks > 0:
                    dict_entry['avg_points_form'] = (
                        cumulative_stats['total_points'] -
                        cumulative_stat_dict_3_weeks_ago['total_points']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_minutes_form'] = (
                        cumulative_stats['total_minutes'] -
                        cumulative_stat_dict_3_weeks_ago['total_minutes']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_goals_scored_form'] = (
                        cumulative_stats['total_goals_scored'] -
                        cumulative_stat_dict_3_weeks_ago['total_goals_scored']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_assists_form'] = (
                        cumulative_stats['total_assists'] -
                        cumulative_stat_dict_3_weeks_ago['total_assists']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_goals_conceded_form'] = (
                        cumulative_stats['total_goals_conceded'] -
                        cumulative_stat_dict_3_weeks_ago['total_goals_conceded']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_clean_sheets_form'] = (
                        cumulative_stats['total_clean_sheets'] -
                        cumulative_stat_dict_3_weeks_ago['total_clean_sheets']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_yellow_cards_form'] = (
                        cumulative_stats['total_yellow_cards'] -
                        cumulative_stat_dict_3_weeks_ago['total_yellow_cards']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_red_cards_form'] = (
                        cumulative_stats['total_red_cards'] -
                        cumulative_stat_dict_3_weeks_ago['total_red_cards']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_saves_form'] = (
                        cumulative_stats['total_saves'] -
                        cumulative_stat_dict_3_weeks_ago['total_saves']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_bps_form'] = (
                        cumulative_stats['total_bps'] -
                        cumulative_stat_dict_3_weeks_ago['total_bps']
                    ) / matches_played_in_last_3_weeks
                    dict_entry['avg_net_transfers_form'] = (
                        cumulative_stats['total_net_transfers'] -
                        cumulative_stat_dict_3_weeks_ago['total_net_transfers']
                    ) / matches_played_in_last_3_weeks
                else:
                    # not played for last 3 weeks. put season averages instead
                    dict_entry['avg_points_form'] = dict_entry['points_per_match_played']
                    dict_entry['avg_minutes_form'] = dict_entry['minutes_per_match_played']
                    dict_entry['avg_goals_scored_form'] = dict_entry['goals_scored_per_match_played']
                    dict_entry['avg_assists_form'] = dict_entry['assists_per_match_played']
                    dict_entry['avg_goals_conceded_form'] = dict_entry['goals_conceded_per_match_played']
                    dict_entry['avg_clean_sheets_form'] = dict_entry['clean_sheets_per_match_played']
                    dict_entry['avg_yellow_cards_form'] = dict_entry['yellow_cards_per_match_played']
                    dict_entry['avg_red_cards_form'] = dict_entry['red_cards_per_match_played']
                    dict_entry['avg_saves_form'] = dict_entry['saves_per_match_played']
                    dict_entry['avg_bps_form'] = dict_entry['bps_per_match_played']
                    dict_entry['avg_net_transfers_form'] = dict_entry['net_transfers_per_match_played']

                dict_entry['last_season_points_per_minutes'] = float(value[i - 1]['last_season_points_per_minutes'])
                dict_entry['team_points_per_match_last_season'] = float(value[i - 1]['team_points_per_match_last_season'])
                dict_entry['opponent_points_per_match_last_season'] = float(value[i - 1]['opponent_points_per_match_last_season'])
                dict_entry['price'] = float(value[i - 1]['price'])
                dict_entry['price_change_form'] = float(value[i - 1]['price']) - float(value[i - 3]['price'])
                dict_entry['team_points_per_match'] = float(value[i - 1]['team_points_per_match'])
                dict_entry['team_goals_scored_per_match'] = float(value[i - 1]['team_goals_scored_per_match'])
                dict_entry['team_goals_conceded_per_match'] = float(value[i - 1]['team_goals_conceded_per_match'])
                dict_entry['is_at_home'] = 1 if value[i]['is_at_home'] else 0
                dict_entry['opponent_points_per_match'] = float(value[i - 1]['opponent_points_per_match'])
                dict_entry['opponent_goals_scored_per_match'] = float(value[i - 1]['opponent_goals_scored_per_match'])
                dict_entry['opponent_goals_conceded_per_match'] = float(value[i - 1]['opponent_goals_conceded_per_match'])

                # print str(key)
                data_dict[key]['data'][i] = {
                    'X': dict_entry,
                    'Y': {'points_scored': float(value[i]['points'])}
                }

    # adding entries for 2016 season
    id_prefix = 16000
    f_path = os.path.join(SCRIPT_DIR, 'fpl_2016_data.json')
    with open(f_path, 'r') as f:
        data_2016 = json.load(f)

    master_url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
    response = urllib.urlopen(master_url)
    fpl_master_data = json.loads(response.read())
    master_elements = sorted(fpl_master_data['elements'], key=lambda k: int(k['id']))

    # to create team name dict
    team_dict = fpl_master_data['teams']
    team_name_dict = {}
    for team in team_dict:
        team_name_dict[team['code']] = team['name']

    table_data_dict = {}

    for player_dict in data_2016['elements']:
        player_data = player_dict['player_data']
        id = int(player_dict['id'])
        element = player_dict
        name = '%s %s' % (strip_accents(element['first_name']), strip_accents(element['second_name']))
        team_id = element['team_code']
        player_type = int(element['element_type'])
        position = [entry['singular_name'] for entry in player_type_dict if int(entry['id']) == player_type][0]
        coming_fixture = player_data['fixtures'][0]
        coming_week = int(coming_fixture['event'])
        meta_data = {
            'name': name,
            'position': position,
            'team': team_name_dict[team_id],
        }
        X_and_Y = []

        data_dict_per_week = defaultdict(dict)
        key = id_prefix + int(id)
        data_dict[key] = {
            'meta': meta_data,
            'data': data_dict_per_week,
        }

        for week in range(3, coming_week):
            # find table position
            table_key = week
            if table_key in table_data_dict:
                table_data = table_data_dict[table_key]
            else:
                matchday = week - 1
                try:
                    league_table_url = 'http://api.football-data.org/v1/competitions/426/leagueTable/?matchday=%s' % matchday
                except:
                    print('Sorry. Something went wrong. Try again later')
                    return ["Sorry. Something went wrong. Try again later"]
                response = urllib.urlopen(league_table_url)
                table_data = json.loads(response.read())
                table_data_dict[table_key] = table_data
            X_pos_y = generate_X_Y_dict(
                id,
                player_data=player_data,
                table_data=table_data,
                fpl_master_data=master_elements,
                fixture_number=week,
            )
            if (len(X_pos_y) != 3):
                # print(X_pos_y[0])
                continue
            X, position, y = X_pos_y
            X_and_Y = {
                'X': X,
                'Y': {'points_scored': y},
            }
            # print(X)
            data_dict[key]['data'][week] = X_and_Y

    f_path = os.path.join(SCRIPT_DIR, 'player_ml_data.json')
    with open(f_path, 'w') as f:
        f.write(json.dumps(data_dict))


def load_dataset(position='midfielder'):
    f_path = os.path.join(SCRIPT_DIR, 'player_ml_data.json')
    with open(f_path, 'r') as f:
        data_dict = json.load(f)

    X_legend = None
    Y_legend = None
    X_list = []
    Y_list = []
    first = True
    # filtered_dict = {k: v for k, v in data_dict.iteritems() if v['meta']['position'].lower() == position.lower()}
    for id, values in data_dict.iteritems():
        if values['meta']['position'].lower() == position.lower():
            for week, data in values['data'].iteritems():
                if not X_legend:
                    X_legend = [k for k, v in data['X'].iteritems()]
                    X_legend = sorted(X_legend)
                    print(X_legend)
                    # print([v for k, v in data['X'].iteritems()])
                if not Y_legend:
                    Y_legend = [k for k, v in data['Y'].iteritems()][0]

                X_l = []
                for i in range(0, len(data['X'])):
                    X_l.append(data['X'][X_legend[i]])
                if first:
                    print('id: %s' % id)
                    print(values['meta']['name'])
                    print('week: %s' % week)
                    print(X_l)
                X_list.append(X_l)
                Y_list.append(data['Y'][Y_legend])
                if first:
                    first = False
    # # write X_legend to file
    legend_path = os.path.join(SCRIPT_DIR, 'X_legend.json')
    with open(legend_path, 'w') as f:
        f.write(json.dumps(X_legend))
    return X_list, Y_list, X_legend, Y_legend


def create_dataset_ssv_file(position='midfielder'):
    X, Y, X_legend, Y_legend = load_dataset(position=position)
    filename = 'fpl_%ss.ssv' % (position.lower())
    with open(filename, 'w') as f:
        for sample in zip(X, Y):
            X_string = ' '.join(map(str, sample[0]))
            f.write(X_string + ' ' + str(sample[1]) + '\n')


# one time functions
def create_fixture_history_csv():
    with open('england_master.csv', 'r') as read_file, open('england_2013-2015_fixtures.csv', 'w') as write_file:
        wr = csv.writer(write_file, quoting=csv.QUOTE_ALL)
        wr.writerow(["date", "season", "home", "visitor", "home_goals", "visitor_goals"])
        # print len(read_file.readlines())
        # next(read_file)
        for line in read_file:
            entries = [entry.strip().strip('"') for entry in line.split(',')]
            # print "season %s - division %s - tier %s" % (entries[1], entries[7], entries[8])
            try:
                if int(entries[1]) >= 2013 and int(entries[7]) == 1 and int(entries[8]) == 1:
                    write_list = [entries[0], entries[1], entries[2], entries[3], entries[5], entries[6]]
                    wr.writerow(write_list)
            except ValueError:
                pass

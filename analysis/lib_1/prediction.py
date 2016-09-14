import numpy as np
import urllib
import json
import os
import random
# import h5py
# from sklearn.externals import joblib
from difflib import SequenceMatcher
import unicodedata
# from keras.models import load_model
from keras.models import model_from_json

from create_dataset import get_average_points_per_minute_dict, load_dataset

SCRIPT_DIR = os.path.dirname(__file__)


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

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


def get_fpl_player_id(last_name, first_name=None, fpl_master_data=None):
    """returns fpl player id
    """
    if not fpl_master_data:
        master_url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
        response = urllib.urlopen(master_url)
        fpl_master_data = json.loads(response.read())

    matched_ids = []
    similarity_cutoff = 0.80

    last_name = last_name.lower().strip()

    for player in fpl_master_data['elements']:
        second_name = strip_accents(player['second_name'])
        words = second_name.split()
        match_ratio = 0.0
        for name in words:
            this_match_ratio = SequenceMatcher(None, last_name, name.lower().strip()).ratio()
            if this_match_ratio > match_ratio:
                match_ratio = this_match_ratio
        if match_ratio > similarity_cutoff:
            matched_ids.append(int(player['id']))
    return matched_ids


def generate_X_dict(player_id, player_data=None, table_data=None, fpl_master_data=None):
    """generates a dictionary with features for a player
    """
    if not player_data:
        player_url = 'https://fantasy.premierleague.com/drf/element-summary/%s' % (player_id)
        response = urllib.urlopen(player_url)
        player_data = json.loads(response.read())

    if not fpl_master_data:
        master_url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
        response = urllib.urlopen(master_url)
        fpl_master_data = json.loads(response.read())

    player_X_dict = {}

    player_gen_data = None
    for element in fpl_master_data['elements']:
        # print element
        if int(element['id']) == player_id:
            player_gen_data = element
            break
    if not player_gen_data:
        print "no player gen data"
        return "Sorry. Something went wrong. Try again later"

    player_type = int(player_gen_data['element_type'])
    player_type_dict = fpl_master_data['element_types']

    position = [entry['singular_name'] for entry in player_type_dict if int(entry['id']) == player_type][0]

    fixture = player_data['fixtures'][0]
    is_home = fixture['is_home']
    week = int(fixture['event'])

    team_id = int(fixture['team_h']) if is_home else fixture['team_a']
    opp_team_id = int(fixture['team_a']) if is_home else fixture['team_h']
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
    player_X_dict['price'] = float(player_gen_data['now_cost']) / 10

    history = player_data['history']
    if len(history) < 3:
        print "not enough matches played, cannot predict"
        return "not enough matches played, cannot predict"
    history = sorted(history, key=lambda k: k['round'])
    history = history[:week - 1]
    if len(history) < 3:
        print "not enough matches played, cannot predict"
        return "not enough matches played, cannot predict"

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
        return "not enough matches played for prediction"
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
    ) / matches_played

    if not table_data:
        league_table_url = 'http://api.football-data.org/v1/competitions/426/leagueTable'
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
        return "Sorry. Something went wrong. Try again later"

    team_matches_played = int(team_table_data['playedGames'])
    opp_team_matches_played = int(opp_team_table_data['playedGames'])

    player_X_dict['team_goals_scored_per_match'] = float(team_table_data['goals']) / team_matches_played
    player_X_dict['team_goals_conceded_per_match'] = float(team_table_data['goalsAgainst']) / team_matches_played
    player_X_dict['team_points_per_match'] = float(team_table_data['points']) / team_matches_played

    player_X_dict['opponent_goals_scored_per_match'] = float(opp_team_table_data['goals']) / opp_team_matches_played
    player_X_dict['opponent_goals_conceded_per_match'] = float(opp_team_table_data['goalsAgainst']) / opp_team_matches_played
    player_X_dict['opponent_points_per_match'] = float(opp_team_table_data['points']) / opp_team_matches_played

    return player_X_dict, position


def predict_next_round_points(player_id, fpl_master_data=None, player_data=None, table_data=None):
    """predicts points to be scored by player (by fpl id) in the next round
    """
    X_dict_return_val = generate_X_dict(
        player_id,
        fpl_master_data=fpl_master_data,
        player_data=player_data,
        table_data=table_data
    )
    try:
        X_dict, position = X_dict_return_val
    except (TypeError, ValueError):
        return X_dict_return_val

    # print X_dict
    # players_data = load_dataset(position='forward')
    legend_path = os.path.join(SCRIPT_DIR, 'X_legend.json')
    with open(legend_path, 'r') as f:
        X_legend = json.load(f)
    # X_legend = players_data[2]
    # print X_legend
    X_list = [0.0] * len(X_legend)
    for key, value in X_dict.iteritems():
        # print key
        # print "%s = %s" % (key, value)
        index = X_legend.index(key)
        X_list[index] = value

    X = np.array([X_list])
    # load model
    position = position.lower()
    '''
    filepath = 'dumps/keras_%ss/keras_%ss.pkl' % (position, position)
    filepath = os.path.join(SCRIPT_DIR, filepath)
    model = joblib.load(filepath)
    '''
    model_path = os.path.join(SCRIPT_DIR, 'dumps/keras_%ss/keras_%ss.json' % (position, position))
    weights_filepath = os.path.join(SCRIPT_DIR, 'dumps/keras_%ss/weights.h5' % (position))
    mean_filepath = os.path.join(SCRIPT_DIR, 'dumps/keras_%ss/mean.json' % (position))
    scale_filepath = os.path.join(SCRIPT_DIR, 'dumps/keras_%ss/scale.json' % (position))

    # model = load_model(model_path)
    with open(model_path) as f:
        model_json = f.read()
    with open(mean_filepath) as f:
        means = json.load(f)
    with open(scale_filepath) as f:
        scales = json.load(f)
    # print model_json
    # print means
    # print scales
    model = model_from_json(model_json)
    model.load_weights(weights_filepath)
    # weights = model.get_weights()
    # print "weights - "
    # print weights[0:2]
    model.compile(loss='mean_squared_error', optimizer='adam')
    # print "original X"
    # print X
    X_transformed = (X - means) / scales
    # print "transformed X"
    # print X_transformed
    prediction = model.predict(X_transformed)
    # print "prediction = %s" % prediction
    predicted_points = int(round(prediction[0]))
    return predicted_points


def predict(last_name, first_name=None):
    """returns list of players with matching name and their predicted scores
    """
    master_url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
    response = urllib.urlopen(master_url)
    fpl_master_data = json.loads(response.read())

    league_table_url = 'http://api.football-data.org/v1/competitions/426/leagueTable'
    response = urllib.urlopen(league_table_url)
    table_data = json.loads(response.read())

    id_list = get_fpl_player_id(
        last_name,
        first_name=first_name,
        fpl_master_data=fpl_master_data,
    )
    predictions = []
    if len(id_list) == 0:
        return ["Sorry, I couldn't find any player with last name '%s'. Please try a different name." % last_name]
    for player_id in id_list:
        player_url = 'https://fantasy.premierleague.com/drf/element-summary/%s' % (player_id)
        response = urllib.urlopen(player_url)
        player_data = json.loads(response.read())
        predicted_points = predict_next_round_points(
            player_id,
            fpl_master_data=fpl_master_data,
            player_data=player_data,
            table_data=table_data
        )

        if isinstance(predicted_points, str):
            predictions.append(predicted_points)
            continue

        for player in fpl_master_data['elements']:
            if int(player['id']) == player_id:
                element_type = int(player['element_type'])
                position = fpl_master_data['element_types'][element_type - 1]['singular_name']
                position_short = fpl_master_data['element_types'][element_type - 1]['singular_name_short']
                fixture = player_data['fixtures'][0]
                is_home = fixture['is_home']
                week = int(fixture['event'])
                # find self team
                team_id = int(player['team'])
                team = next((item for item in fpl_master_data['teams'] if item['id'] == team_id))
                team_name = team['name']
                team_shortname = team['short_name']
                # find opposition team
                opp_team_id = int(fixture['team_a']) if is_home else fixture['team_h']
                opp_team = next((item for item in fpl_master_data['teams'] if item['id'] == opp_team_id))
                opp_team_name = opp_team['name']
                opp_team_shortname = opp_team['short_name']
                entry = {
                    'id': player_id,
                    'first_name': player['first_name'],
                    'second_name': player['second_name'],
                    'team_name': team_name,
                    'team_shortname': team_shortname,
                    'position': position,
                    'position_short': position_short,
                    'opp_team_name': opp_team_name,
                    'opp_team_shortname': opp_team_shortname,
                    'predicted_points': predicted_points,
                    'week': week,
                }
                predictions.append(entry)
                break

    return get_prediction_response(predictions)


def get_prediction_response(predictions):
    """returns a response given a prediction dict
    """
    if len(predictions) == 0:
        # no player identified
        return ["Sorry, I couldn't find any player with that last name. Please try a different name."]
    responses = []
    for prediction in predictions:
        # # handle error messages
        if isinstance(prediction, str):
            responses.append(prediction)
            continue

        # # response for identified player
        start_message = random.choice([
            'I think ',
            'I predict ',
            'After careful consideration, I can say that ',
            'It seems that ',
        ])
        if prediction['predicted_points'] > 1:
            plural_string = 's'
        else:
            plural_string = ''
        response = '%s %s %s (%s) of %s will probably score around %d point%s against %s in GW %d' % (
            start_message,
            prediction['first_name'],
            prediction['second_name'],
            prediction['position_short'],
            prediction['team_name'],
            prediction['predicted_points'],
            plural_string,
            prediction['opp_team_name'],
            prediction['week']
        )
        responses.append(response)
    return responses

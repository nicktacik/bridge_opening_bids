from consants import data_dir, zip_file, bid_cyclic
import os
import zipfile
from glob import glob
import re
from data_models import Hand


def unzip_data_files():
    if not os.path.exists(data_dir):
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        zip_ref.extractall()
        zip_file.close()


def create_raw_dataset_for_file(f):
    with open(f, encoding='windows-1251') as pbn_file:
        next_output = {
            'Auction': None,
            'Dealer': None,
            'Deal': None,
            'Vul': None
        }
        output = []
        auction_tracking = False
        for line in pbn_file:
            if auction_tracking:
                if line.startswith('['):
                    auction_tracking = False
                    output.append(next_output)
                    next_output = {}
                else:
                    next_output['Auction'] += line
            elif line.startswith('[Dealer'):
                next_output['Dealer'] = line
            elif line.startswith('[Deal'):
                next_output['Deal'] = line
            elif line.startswith('[Vulnerable'):
                next_output['Vul'] = line
            elif line.startswith('[Auction'):
                next_output['Auction'] = line
                auction_tracking = True

        return output


def load_raw_data():
    unzip_data_files()
    raw_data = [create_raw_dataset_for_file(f) for f in glob(data_dir + '*pbn')]
    raw_data = sum(raw_data, [])  # flatten the data
    return raw_data


def load_clean_data(raw_data=None, max_seat=1):
    # the data structure here will be a list of lists
    # where the outer list is each unique hand in the data set
    # and the inner list is the result (hand object) for each

    if not raw_data:
        raw_data = load_raw_data()

    output = []
    for deal in raw_data:
        cleaned_hands = make_clean_data_from_deal(deal)
        for hand in cleaned_hands:
            if hand.seat <= max_seat:
                try:
                    found_index = len(output) - 1 - next(i for i, recorded_hands in enumerate(reversed(output)) if
                                       hand.same_hand(recorded_hands[0]))
                    output[found_index].append(hand)
                except StopIteration:
                    output.append([hand])
    return output


def make_clean_data_from_deal(deal):
    # deal is a dictionary containing Auction, Dealer, Deal, Vul
    # output is a list of Hand objects
    dealer = deal['Dealer'].split('"')[1]
    deal_start = deal['Deal'].split('"')[1][0]
    vul = deal['Vul'].split('"')[1]
    vul = 'NSEW' if vul == 'All' else vul
    auction = deal['Auction'].replace('\n', ' ')
    auction = re.sub(r'\[.+?\]', '', auction)  # remove everything between square brackets
    auction = re.sub(r'\{.+?\}', '', auction)  # remove everything between curly braces
    auction = re.sub(' +', ' ', auction)  # get rid of duplicate spaces
    auction = auction.strip()
    if not auction:
        return []  # weed out bad data
    bids = auction.split(" ")
    hands = deal['Deal'].split(":")[1].split(" ")
    output = []
    offset = cyclic_relations(deal_start, dealer)
    bid_start = False
    bidder = deal_start

    while not (bid_start or len(output) >= 4):
        hand, bid = hands[offset], bids[len(output)]
        hand = hand.split('"')[0]

        output.append(Hand(hand=hand, bid=bid, seat=len(output), vul=bidder in vul))
        if bid != 'Pass':
            bid_start = True
        else:
            offset = offset + 1 if offset != 3 else 0
            bidder = next_bidder(bidder)

    return output


def cyclic_relations(x, y):
    if bid_cyclic[y] >= bid_cyclic[x]:
        return bid_cyclic[y] - bid_cyclic[x]
    else:
        return 4 + bid_cyclic[y] - bid_cyclic[x]


def next_bidder(x):
    offset = bid_cyclic[x]
    offset = offset + 1 if offset < 3 else 0
    return [k for k, v in bid_cyclic.items() if v == offset][0]

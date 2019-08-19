import numpy as np
from consants import suits


class Hand:

    def __init__(self, hand, vul, seat, bid):
        self.hand = hand
        self.vul = vul
        self.seat = seat
        self.bid = bid

    def __repr__(self):
        return "Hand={0} Bid={1} Seat={2}".format(self.hand, self.bid, self.seat)

    def same_hand(self, hand):
        # check if two hands are the same, ignoring bid
        return self.hand == hand.hand and self.vul == hand.vul and self.seat == hand.seat


class Model:

    def __init__(self):
        pass

    def size(self):
        raise NotImplementedError()

    def encode_hand(self, hand):
        raise NotImplementedError()

    @staticmethod
    def encode_bid(bid):
        if bid.lower() == 'pass':
            code = 0
        else:
            level = int(bid[0])
            suit = suits[bid[1:]]
            code = (level-1)*5 + suit + 1
        out = np.zeros(36)
        out[code] = 1.
        return out

    @staticmethod
    def decode_bid(encoded_bid):
        # allow for either a numpy array or an int to represent the bid
        if not isinstance(encoded_bid, int):
            encoded_bid = np.argmax(encoded_bid)
        if encoded_bid == 0:
            return 'Pass'
        else:
            level = (encoded_bid-1) // 5 + 1
            suit = (encoded_bid-1) % 5
            return str(level) + suits[suit]


class BasicModel(Model):

    def __init__(self):
        super().__init__()
        self.card_to_val = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4,
                            '7': 5, '8': 6, '9': 7, 'T': 8, 'J': 9,
                            'Q': 10, 'K': 11, 'A': 12}

    def encode_hand(self, hand):
        output = np.zeros(self.size())
        for i, suit in enumerate(hand.hand.split('.')):
            for card in suit:
                output[i*13 + self.card_to_val[card]] = 1.0
        output[52] = 1.0 if hand.vul else 0.0

        return output

    def size(self):
        return 53


class AdvancedModel(Model):

    def __init__(self, use_hcp=True, use_dist=True):
        super().__init__()
        self.card_to_val = {'x': 0, 'X': 0, '2': 0, '3': 0, '4': 0, '5': 0, '6': 0,
                            '7': 0, '8': 0, '9': 0, 'T': 1, 'J': 2,
                            'Q': 3, 'K': 4, 'A': 5}
        self.card_to_hcp = {'J': 1.0, 'Q': 2.0, 'K': 3.0, 'A': 4.0}

        self.use_hcp = use_hcp
        self.use_dist = use_dist

    def encode_hand(self, hand):
        voids, singles, doubles, hcp = 0.0, 0.0, 0.0, 0.0
        output = np.zeros(self.size())

        for i, suit in enumerate(hand.hand.split('.')):
            for card in suit:
                hcp += self.card_to_hcp.get(card, 0)
                if self.card_to_val[card] == 0:
                    output[i*6] += 1/8.0  # normalize so that this is 0.25 on average like the other cards
                else:
                    output[i*6 + self.card_to_val[card]] = 1.0

            if len(suit) == 0:
                voids += 1
            elif len(suit) == 1:
                singles += 1
            elif len(suit) == 2:
                doubles += 1

        balanced = (voids == 0 and singles == 0 and doubles <= 1)
        output[24] = 1.0 if hand.vul else 0.0
        output_idx = 25
        if self.use_hcp:
            output[output_idx] = hcp / 52.0  # normalize to be 0.25 on average
            output_idx += 1
        if self.use_dist:
            output[output_idx] = voids
            output[output_idx+1] = singles
            output[output_idx+2] = 1.0 if balanced else 0.0

        return output

    def size(self):
        # cards + vul + hcp + (voids/singles/balanced)
        return 24 + 1 + (1 if self.use_hcp else 0) + (3 if self.use_dist else 0)
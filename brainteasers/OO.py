import random

#region Deck of cards using OO
class Card:
    def __init__(self, suit, val):
        if str(suit).lower() not in ['ace','spade','club','diamond']:
            raise Exception("suit must be in ace, spade, club or diamond")
        else:
            self.val = int(val)
            self.suit = str(suit).lower()

    def __repr__(self):
        return str(self.val)+' of '+str(self.suit)

class Deck:
    def __init__(self):
        self.cards = []
        self.build()

    def build(self):
        for _suit in ['ace','spade','club','diamond']:
            for _val in range(1,14):
                self.cards.append(Card(_suit, _val))

    def __repr__(self):
        res = ''
        for card in self.cards:
            res += card.__repr__() + "\n"
        return res

    def drawCard(self):
        return self.cards.pop()

    def drawCard_rdn(self):
        rand = random.randint(0, len(self.cards))
        return self.cards[rand]

    def shuffle(self):
        for i in range(len(self.cards)-1,0,-1):
            rand = random.randint(0, i)
            # swap cards
            self.cards[i], self.cards[rand] = self.cards[rand], self.cards[i]


ace1 = Card('spade',1)
print(ace1)
deck = Deck()
print(deck)
#endregion

#region Musical Jukebox using OO TODO
#endregion
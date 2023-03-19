"""
Blackjack

The classic card game also known as 21.
"""

import random, sys 

HEARTS = chr(9829)
DIAMONDS = chr(9830)
SPADES = chr(9824)
CLUBS = chr(9827)
BACKSIDE = 'backside'

def main():
    print(
        '''
        Rules:
        Try to get as close to 21 without going over.
        Kings, Queens, and Jack are worth 10 points.
        Aces are worth 1 or 11 points.
        Cards 2 through 10 are worth their face value.
        '''
        )
    
    money = 5000
    while True:

        if money <= 0:
            print("You're broke.")
            print("Good thing you weren't playing with real money.")
            print("Thanks for playing!")
            sys.exit()

        print('Money:', money)
        bet = getBet(money)

        deck = getDeck()
        dealerHand = [deck.pop(), deck.pop()]
        playerHand = [deck.pop(), deck.pop()]

        print('Bet:', bet)
        while True:

            displayHands(playerHand, dealerHand, False)
            print()

            if getHandValue(playerHand) > 21: 
                break 

            move = getMove(playerHand, money - bet)

            if move == 'D':
                additionalBet = getBet(min(bet, (money - bet)))

                bet += additionalBet

                print('Bet increased to {}'.format(bet))
                print('Bet:', bet)

            if move in ('H', 'D'):
                newCard = deck.pop()
                rank, suit = newCard 

                print('You drew a {} of {}'.format(rank, suit))
                playerHand.append(newCard)

                if getHandValue(playerHand) > 21:
                    continue 

            if move in ('S', 'D'):
                break 

        if getHandValue(playerHand) <= 21:

            while getHandValue(dealerHand) < 17:
                print('Dealer hits...')
                dealerHand.append(deck.pop())
                displayHands(playerHand, dealerHand, False)

                if getHandValue(dealerHand) > 21:
                    break 

                input('Press Enter to continue...')
                print('\n\n')

        displayHands(playerHand, dealerHand, True)

        playerValue = getHandValue(playerHand)
        dealerValue = getHandValue(dealerHand)

        if dealerValue > 21:
            print("Dealer busts! You win ${}!".format(bet))
            money += bet 
        elif (playerValue > 21) or (playerValue < dealerValue):
            print('You lost!')
            money -= bet 
        elif playerValue > dealerValue:
            print('You won ${}!'.format(bet))
            money += bet 
        elif playerValue == dealerValue:
            print("It\'s a tie, the bet is returned to you.")

        input('Press Enter to continue...')
        print('\n\n')

def getBet(maxBet):
    """Ask the player how much they want to bet for this round."""

    while True: 
        print('How much do you bet? (1-{}, or QUIT)'.format(maxBet))
        bet = input('> ').upper().strip()
        if bet == 'QUIT':
            print('Thanks for playing!')
            sys.exit() 

        if not bet.isdecimal():
            continue 

        bet = int(bet)
        if 1 <= bet <= maxBet: 
            return bet 
    
def getDeck():
    """Return a list of (rank, suit) tuples for all 52 cards."""

    deck = []
    for suit in (HEARTS, DIAMONDS, SPADES, CLUBS):
        for rank in range(2, 11):
            deck.append((str(rank), suit))
        for rank in ('J', 'Q', 'K', 'A'):
            deck.append((rank, suit)) 

    random.shuffle(deck) 
    return deck 

def displayHands(playerHand, dealerHand, showDealerHand):
    """Show the player's and dealer's cards. Hide the dealer's first card if showDealerHand is False."""

    print()
    if showDealerHand:
        print('DEALER:', getHandValue(dealerHand))
        displayCards(dealerHand)
    else:
        print('DEALER: ???')
        displayCards([BACKSIDE] + dealerHand[1:])

    print('PLAYER:', getHandValue(playerHand))
    displayCards(playerHand)

def getHandValue(cards):
    """Return the value of the cards. Face cards are worth 10, aces are worth 11 or 1."""
    value = 0
    numberOfAces = 0

    for card in cards:
        rank = card[0]
        if rank == 'A':
            numberOfAces += 1
        elif rank in ('K', 'Q', 'J'):
            value += 10
        else:
            value += int(rank)

    value += numberOfAces
    for i in range(numberOfAces):
        if value + 10 <= 21:
            value += 10 

    return value 

def displayCards(cards):
    """Display all the cards in the cards list."""
    rows = ['', '', '', '', '']  # The text to display on each row.

    for i, card in enumerate(cards):
        rows[0] += ' ___  '  # Print the top line of the card.
        if card == BACKSIDE:
            rows[1] += '|## | '
            rows[2] += '|###| '
            rows[3] += '|_##| '
        else:
            rank, suit = card  # The card is a tuple data structure.
            rows[1] += '|{} | '.format(rank.ljust(2))
            rows[2] += '| {} | '.format(suit)
            rows[3] += '|_{}| '.format(rank.rjust(2, '_'))

    for row in rows:
        print(row)

def getMove(playerHand, money):
    """Asks the player for their move, and returns 'H' for hit, 'S' for stand, and 'D' for double down."""

    while True:
        moves = ['(H)it', '(S)tand']

        if len(playerHand) == 2 and money > 0:
            moves.append('(D)ouble down')

        movePrompt = ', '.join(moves) + '> '
        move = input(movePrompt).upper()
        if move in ('H', 'S'):
            return move 
        if move == 'D' and '(D)ouble down' in moves: 
            return move 
        
if __name__ == '__main__':
    main()

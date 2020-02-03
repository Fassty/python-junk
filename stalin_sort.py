#!/usr/bin/env python
import random
"""
This is Stalin Sort
It sorts a list in descending order in linear time
It does so by simply erradicating the elments that are not in order
Because we do not want those disruptive elements
in our beautiful mother Russia
"""

def bring_order_to_party(party):
    disruptors = []
    stalin = party[0]
    for _id, comrade in enumerate(party[1:]):
        if comrade < stalin:
            stalin = comrade
        else:
            disruptors.append(_id + 1)

    party = send_to_vacation(party, disruptors)
    return party

def send_to_vacation(party, disruptors):
    party = [comrade for number, comrade in enumerate(party) if number not in disruptors]
    return party

def generate_party(size=10):
    party = []
    for i in range(size):
        party.append(random.randint(1,100))
    return party

party = generate_party(20)
print(party)
print("Bringing order to a party...")
print(bring_order_to_party(party))
print("Traitors were sucessfully taken care of. Glory to mother Russia.")

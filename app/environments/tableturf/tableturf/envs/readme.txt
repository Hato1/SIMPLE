ToDo: Poetry env.

Input:
	Board of maximum possible dimensions 19x25. Many of these spots will be neutral borders depending on the map.
	All 175 available cards
		deck?


Variables:
	Chosen map.
	Opponents deck?

Actions:
	Form a 15 card deck from 175.
	Mulligan
	For 12 turns either:
	    For each point on the 19x25 grid, every card (mapped to the bottom left tile) can be placed in all 4 rotations and could have special active.
	    Pass by discarding one of the 4 cards in hand.

Gameplay:
	First 15 actions:
		Choose a Card (Card)
	Mulligan: (Bool)
	Final 12 actions:
		Choose a Card (Card)
		Activate special (Bool)
		Rotation (Direction 0-3)
		X_placement (int 0-24)
		Y_placement (int 0-18)


Legal actions:
	Deck creation:
		Same card cannot be picked twice.
		Order that cards are picked does not matter.
	Mulligan:
		Only accepting or rejecting the mulligan.
	Playing game:
		Chosen card must be in hand.
		Special can only be true if it can be afforded.
		Rotation, X_placement and Y_placement can only overlap empty tiles.
		If special is activated, yours and opponents tiles can also overlap.


AI Strategies to consider:
Recurrent neural network (cool).
Decision Tree.
Random Forest.

Ideas:
When checking legal actions: Get adjacencies (including diagonals) to numpy array by growing it via convolution with a 3x3 matrix
Specify colour ranges more easily by converting to HSV first and using a Hue range. This surely beats complicated sets of RGB ranges.
Get all legal placements of card shape on board by correlating it (numpy correlation)
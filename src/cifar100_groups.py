"""CIFAR-100 superclass / fine-class metadata and the six similar-group pairs.

This project treats each superclass as a 5-way classification dataset and studies
expert transfer across semantically similar superclass pairs.
"""

from __future__ import annotations

from collections import OrderedDict

CIFAR100_FINE_LABELS = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "computer_keyboard", "lamp", "lawn_mower", "leopard",
    "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree",
    "pear", "pickup_truck", "pine_tree", "plain", "plate", "poppy", "porcupine",
    "possum", "rabbit", "raccoon", "ray", "road", "rocket", "rose",
    "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake",
    "spider", "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor", "train", "trout",
    "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]

GROUP_TO_FINE_LABELS = OrderedDict(
    {
        "aquatic_mammals": ["beaver", "dolphin", "otter", "seal", "whale"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "flowers": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "food_containers": ["bottle", "bowl", "can", "cup", "plate"],
        "fruit_and_vegetables": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "household_electrical_devices": ["clock", "computer_keyboard", "lamp", "telephone", "television"],
        "household_furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "insects": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "large_carnivores": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large_manmade_outdoor_things": ["bridge", "castle", "house", "road", "skyscraper"],
        "large_natural_outdoor_scenes": ["cloud", "forest", "mountain", "plain", "sea"],
        "large_omnivores_and_herbivores": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "medium_sized_mammals": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "small_mammals": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "trees": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "vehicles_1": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles_2": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }
)

LABEL_TO_INDEX = {name: idx for idx, name in enumerate(CIFAR100_FINE_LABELS)}
GROUP_TO_FINE_INDICES = {
    group: [LABEL_TO_INDEX[name] for name in fine_names]
    for group, fine_names in GROUP_TO_FINE_LABELS.items()
}

SIMILAR_GROUP_PAIRS = [
    ("vehicles_1", "vehicles_2"),
    ("aquatic_mammals", "fish"),
    ("flowers", "trees"),
    ("insects", "non_insect_invertebrates"),
    ("small_mammals", "medium_sized_mammals"),
    ("large_carnivores", "large_omnivores_and_herbivores"),
]

PAIR_NAME_MAP = {
    ("vehicles_1", "vehicles_2"): "vehicles1_to_vehicles2",
    ("aquatic_mammals", "fish"): "aquatic_mammals_to_fish",
    ("flowers", "trees"): "flowers_to_trees",
    ("insects", "non_insect_invertebrates"): "insects_to_non_insect_invertebrates",
    ("small_mammals", "medium_sized_mammals"): "small_mammals_to_medium_sized_mammals",
    ("large_carnivores", "large_omnivores_and_herbivores"): "large_carnivores_to_large_omnivores_and_herbivores",
}


def normalize_group_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def canonical_group_name(name: str) -> str:
    normalized = normalize_group_name(name)
    for group_name in GROUP_TO_FINE_LABELS:
        if normalized == group_name:
            return group_name
    aliases = {
        "vehicles1": "vehicles_1",
        "vehicles2": "vehicles_2",
        "non_insect_invertebrates": "non_insect_invertebrates",
        "large_omnivores_and_herbivores": "large_omnivores_and_herbivores",
        "small_mammals": "small_mammals",
        "medium_sized_mammals": "medium_sized_mammals",
    }
    if normalized in aliases:
        return aliases[normalized]
    raise KeyError(f"Unknown CIFAR-100 group name: {name}")

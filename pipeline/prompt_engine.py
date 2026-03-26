"""
Prompt Engine — Systematic prompt generation for photorealistic synthetic images.

Phase A: One strong baseline prompt per class.
Phase B: 20 prompts per class, varying across 5 dimensions:
    1. Scene / Environment
    2. Viewpoint / Camera angle
    3. Lighting condition
    4. Disturbance (occlusion, blur, noise)
    5. Object state / quantity / arrangement
"""

import random
from itertools import product

# ─── Variation Dimensions ────────────────────────────────────────────────────

SCENES = {
    "kitchen_item": [
        "on a messy granite kitchen countertop cluttered with other dishes, bottles, and utensils",
        "on a wooden dining table in a busy restaurant with people's arms and other plates visible",
        "in a crowded dish rack next to a dirty sink with sponges and soap bottles",
        "on a stained white marble surface in a lived-in kitchen with crumbs and spills",
        "on a wrinkled picnic blanket outdoors with grass, ants, and other food items scattered around",
        "on a scratched cafeteria tray with other food containers and a napkin dispenser nearby",
        "on a rustic farmhouse table with a wrinkled tablecloth, bread crumbs, and scattered cutlery",
        "on a cluttered office desk next to a keyboard, papers, and a water bottle",
        "in a dimly lit bar counter with bottles, glasses, and coasters everywhere",
    ],
    "food": [
        "on a ceramic plate surrounded by other dishes, a glass of water, and cutlery on a dining table",
        "on a wooden cutting board in a messy kitchen with other vegetables, a knife, and peels nearby",
        "in a crowded street food market stall with other food items, price tags, and a vendor's hands visible",
        "on a restaurant table cluttered with menus, napkins, condiment bottles, and other people's plates",
        "on a checkered picnic blanket in a park with other snacks, drinks, and a person's hand reaching",
        "on a supermarket shelf crammed between other products with price stickers and fluorescent glare",
        "held in a person's hand outdoors with the background showing a busy street and other pedestrians",
        "on a kitchen counter next to a toaster, a bowl of fruit, and a coffee cup",
        "on a school lunch tray with a milk carton, an apple, and a wrapped sandwich",
    ],
    "furniture": [
        "in a lived-in living room with magazines on the floor, a remote on the cushion, and shoes by the door",
        "in a messy college dorm room with books, clothes draped over chairs, and posters on the wall",
        "in a busy office space with other desks, monitors, cables on the floor, and a coat on the chair",
        "in a hotel room with luggage on the floor, a lamp on the nightstand, and curtains partially drawn",
        "in a vintage apartment with scuffed wooden floors, a rug, stacked books, and house plants",
        "in a showroom with other furniture pieces, price tags, and reflections in a nearby mirror",
        "in a cozy cabin with warm wood tones, a blanket draped over, and a steaming mug on a side table",
        "in a small apartment with toys on the floor, a TV on, and a dining table in the background",
        "in a waiting room with fluorescent lights, other chairs, magazines on a table, and scuff marks on the floor",
    ],
}

VIEWPOINTS = [
    "photographed straight-on at eye level",
    "shot from a slightly elevated 45-degree angle",
    "captured from a top-down bird's eye view",
    "taken from a low angle looking upward",
    "photographed from a 3/4 side angle",
    "captured with a close-up macro lens",
    "shot from a wide angle showing the full environment",
]

LIGHTING = [
    "in bright natural daylight from a window",
    "under warm tungsten indoor lighting",
    "with soft diffused overcast lighting",
    "with dramatic side lighting casting shadows",
    "under harsh fluorescent overhead lights",
    "in golden hour sunset lighting",
    "in dim ambient evening lighting with some shadows",
]

DISTURBANCES = [
    "with slight camera shake and minor focus softness like a casual phone photo",
    "with noticeable motion blur on the main subject as if someone bumped the table",
    "with the main object partially occluded by a hand, another object, or a person walking past",
    "with shallow depth of field where parts of the object are out of focus",
    "with visible lens distortion, chromatic aberration at the edges, and slight barrel effect",
    "with a very cluttered and busy background that partially distracts from the subject",
    "with slight overexposure and blown-out highlights near a window or light source",
    "with underexposure and visible image noise/grain as if taken in low light with a phone",
    "with another object partially blocking the view, creating foreground occlusion",
    "with reflections and glare from a glass surface or window nearby",
    "with jpeg compression artifacts and slight color banding visible",
]

OBJECT_STATES = {
    "wine glass": [
        "a wine glass half-filled with red wine next to a bottle and a bowl of olives",
        "two wine glasses on a dinner table with plates of food, napkins, and a candle",
        "a wine glass tipped on its side spilling wine on a tablecloth with a plate and fork nearby",
        "a wine glass with condensation next to a water pitcher and a cutting board with cheese",
        "a dirty wine glass in a sink with other dishes, cups, and soapy water",
    ],
    "cup": [
        "a coffee cup with steam next to a laptop, phone, and scattered papers on a desk",
        "a tea cup on a saucer beside a teapot, sugar bowl, and a spoon on a tray",
        "a colorful mug on a kitchen counter next to a toaster, banana, and crumby plate",
        "a paper coffee cup on a park bench next to a half-eaten sandwich and a bag",
        "multiple cups of different sizes clustered on a drying rack with plates and bowls",
    ],
    "fork": [
        "a fork on a plate with leftover food, next to a knife and a crumpled napkin",
        "a fork with food residue on the tines lying across a bowl of pasta with a glass nearby",
        "two forks and a spoon on a messy table with plates, salt shaker, and bread crumbs",
        "a plastic fork stuck in a takeout container next to another container and a drink cup",
        "a fork partially hidden under a napkin next to a plate of cake and a coffee cup",
    ],
    "knife": [
        "a butter knife on a plate edge next to toast, a jar of jam, and a cup of coffee",
        "a steak knife with a wooden handle next to a plate of steak with a fork and a glass of wine",
        "a dinner knife next to a folded napkin, a fork, and a bowl of soup on a placemat",
        "a knife on a cutting board with sliced bread, cheese, and a bowl of fruit nearby",
        "a table knife partially under a plate next to scattered food crumbs and a water glass",
    ],
    "spoon": [
        "a spoon resting in a bowl of soup next to a plate of bread and a glass of water",
        "a wooden spoon with sauce next to a pot, scattered ingredients, and a messy stovetop",
        "a dessert spoon next to a slice of cake on a plate with a fork and a coffee cup",
        "a spoon sticking out of a yogurt cup next to a banana peel and a granola bar wrapper",
        "multiple spoons and forks jumbled together in a utensil drawer with other kitchen tools",
    ],
    "bowl": [
        "a bowl of salad next to a glass of water, bread basket, and salt and pepper shakers",
        "a chipped porcelain bowl on a counter next to a cutting board, knife, and vegetable scraps",
        "a wooden bowl of mixed fruit on a table with a newspaper, coffee cup, and reading glasses",
        "stacked bowls on a shelf next to plates, cups, and a jar of sugar",
        "a bowl of cereal with a spoon in it next to a milk carton and an orange on a kitchen table",
    ],
    "banana": [
        "a ripe banana with brown spots next to an apple, a bowl, and a cup on a kitchen counter",
        "a bunch of bananas in a fruit bowl with oranges and apples, near a knife and cutting board",
        "a peeled banana on a plate next to a bowl of cereal, a spoon, and a glass of juice",
        "a banana partially hidden behind a water bottle on a cluttered office desk with papers",
        "an overripe banana with dark spots on a grocery bag next to bread and canned food",
    ],
    "apple": [
        "a red apple on a school desk next to books, a pencil case, and a water bottle",
        "a green apple in a fruit bowl with bananas and oranges on a kitchen table near a coffee cup",
        "a half-eaten apple on a messy desk next to a keyboard, phone, and crumpled papers",
        "three apples on a grocery store display partially blocked by a person's shopping basket",
        "an apple next to a sandwich on a cafeteria tray with a milk carton and a napkin",
    ],
    "sandwich": [
        "a club sandwich on a plate next to a bowl of soup, a glass of soda, and a fork",
        "a grilled cheese with melted cheese oozing on a cutting board next to a knife and chips",
        "a sub sandwich in paper on a restaurant table with condiment bottles, napkins, and a drink cup",
        "a half-eaten sandwich on a park bench next to a coffee cup, bag, and a pigeon nearby",
        "a sandwich on a tray next to fries, ketchup packets, and a crumpled napkin in a fast food place",
    ],
    "orange": [
        "an orange on a kitchen counter next to a knife, a cutting board, bananas, and a bowl",
        "a halved orange on a plate next to a glass of juice, a spoon, and toast on another plate",
        "oranges in a mesh bag on a grocery cart with other produce items and a shopping list",
        "a peeled orange with segments on a plate next to a napkin and a cup of tea on a coffee table",
        "an orange partially hidden behind a water bottle and a stack of books on a desk",
    ],
    "broccoli": [
        "broccoli florets on a plate next to grilled chicken, rice, and a fork on a dinner table",
        "steamed broccoli in a bowl next to other side dishes, a glass of water, and salt shaker",
        "broccoli on a cutting board with a knife, scattered carrot peels, and other chopped vegetables",
        "roasted broccoli on a baking sheet with other roasted vegetables and tongs",
        "a head of broccoli in a grocery bag next to carrots, onions, and a receipt",
    ],
    "carrot": [
        "carrots with green tops on a cutting board next to a knife, celery, and a bowl of hummus",
        "sliced carrot rounds in a pot of stew with other vegetables and a wooden spoon",
        "a peeled carrot next to a peeler, carrot shavings, and other vegetables on a messy counter",
        "baby carrots in a bowl next to ranch dip, celery sticks, and a plate on a party table",
        "carrots in a grocery bag with other produce, partially spilling onto a kitchen counter",
    ],
    "hot dog": [
        "a hot dog with mustard in a bun on a paper plate next to fries, a drink cup, and ketchup",
        "two hot dogs on a tray at a ballpark with a beer cup, napkins, and a scoreboard blurry behind",
        "a grilled hot dog on a grill with other sausages, tongs, and smoke rising",
        "a loaded hot dog with toppings next to another hot dog, chips, and a soda can on a table",
        "a hot dog in a wrapper on a street vendor's cart with condiment bottles and other food items",
    ],
    "pizza": [
        "a pizza slice on a paper plate next to a soda can, napkins, and other slices in an open box",
        "a whole pizza in a box on a coffee table with a remote control, cups, and a phone",
        "a pizza slice being pulled by a hand with stretchy cheese, other slices and plates around",
        "a pizza on a wooden board in a restaurant with a beer glass, salad bowl, and other tables behind",
        "leftover pizza slices in a greasy box next to crumpled napkins and an empty bottle",
    ],
    "donut": [
        "a glazed donut on a napkin next to a cup of coffee, a phone, and sugar packets on a cafe table",
        "a chocolate donut with a bite taken out on a desk next to a keyboard and a coffee mug",
        "a box of assorted donuts on a break room table with coffee cups, a paper towel roll, and chairs",
        "a powdered sugar donut on a plate with sugar scattered around, next to a glass of milk",
        "a donut in a paper bag next to a coffee cup on a car dashboard with the windshield visible",
    ],
    "cake": [
        "a slice of chocolate cake on a plate with a fork, next to coffee cups and a candle on a table",
        "a birthday cake with messy frosting and partially melted candles, plates and a knife nearby",
        "a cupcake on a desk next to papers, a laptop screen, and a half-empty water bottle",
        "a partially sliced cheesecake on a stand with dessert plates, forks, and crumbs scattered",
        "a slice of cake on a paper plate at an outdoor party with cups, decorations, and chairs around",
    ],
    "chair": [
        "a wooden dining chair at a table with plates, cups, and a person's bag hanging on the back",
        "an office chair at a cluttered desk with monitors, papers, a coffee cup, and cables",
        "a folding chair on a patio next to a table with drinks, a potted plant, and other chairs",
        "a vintage armchair in a corner with a book on the seat, a lamp beside it, and a rug underneath",
        "multiple chairs around a messy dining table with plates, glasses, and leftover food",
    ],
    "couch": [
        "a gray sofa with messy throw pillows, a blanket, a remote, and a phone on the cushions",
        "a leather couch in a living room with a coffee table, books, cups, shoes on the floor, and a TV on",
        "a sectional sofa with a laptop on it, a person's legs visible, pillows on the floor, and a lamp",
        "a loveseat in a small apartment with clothes draped over the arm, a cat, and a side table with clutter",
        "a couch at a waiting room with other chairs, magazines on a table, and scuffed flooring",
    ],
    "potted plant": [
        "a succulent on a windowsill next to other small plants, a watering can, and dusty window",
        "a large monstera plant next to a couch with a coffee table, books, and a cup in the background",
        "a hanging plant with trailing vines above a desk with a monitor, keyboard, and coffee mug",
        "a cactus on a cluttered desk next to a lamp, phone, pens, and sticky notes",
        "multiple potted plants of different sizes on a shelf with books, picture frames, and a clock",
    ],
    "bed": [
        "a queen bed with rumpled sheets, a phone on the pillow, a nightstand with a lamp and water glass",
        "an unmade bed with clothes thrown on it, shoes on the floor, and a cluttered nightstand",
        "a bed with a laptop on it, pillows against the headboard, a charger cable, and a mug on the nightstand",
        "a twin bed in a dorm room with a desk, chair, books, posters on the wall, and a backpack on the floor",
        "a bed with a cat sleeping on it, a book left open, a glass on the nightstand, and curtains half-drawn",
    ],
}

# ─── Category Mapping ────────────────────────────────────────────────────────

CLASS_CATEGORY = {
    "wine glass": "kitchen_item",
    "cup": "kitchen_item",
    "fork": "kitchen_item",
    "knife": "kitchen_item",
    "spoon": "kitchen_item",
    "bowl": "kitchen_item",
    "banana": "food",
    "apple": "food",
    "sandwich": "food",
    "orange": "food",
    "broccoli": "food",
    "carrot": "food",
    "hot dog": "food",
    "pizza": "food",
    "donut": "food",
    "cake": "food",
    "chair": "furniture",
    "couch": "furniture",
    "potted plant": "furniture",
    "bed": "furniture",
}


# ─── Prompt Assembly ─────────────────────────────────────────────────────────

QUALITY_SUFFIX = (
    "Realistic candid photograph that looks like it was taken by a regular person "
    "with a smartphone or consumer camera, NOT a professional studio photo. "
    "Include natural imperfections: slight color cast, minor noise/grain, "
    "not perfectly composed, real-world messiness and clutter. "
    "Multiple objects should be visible in the scene, some partially occluding each other. "
    "The image should look like it belongs in the COCO dataset — a natural everyday scene "
    "captured casually, not a product photo or stock image. "
    "No watermarks, no text overlays, no artistic filters."
)


def build_prompt(class_name: str, scene: str, viewpoint: str,
                 lighting: str, disturbance: str, object_state: str) -> str:
    """Assemble a full generation prompt from variation components."""
    return (
        f"{object_state}, {scene}, {viewpoint}, {lighting}, {disturbance}. "
        f"{QUALITY_SUFFIX}"
    )


def get_baseline_prompt(class_name: str) -> str:
    """Phase A: one strong baseline prompt per class."""
    category = CLASS_CATEGORY[class_name]
    scene = SCENES[category][0]
    viewpoint = VIEWPOINTS[0]
    lighting = LIGHTING[0]
    disturbance = DISTURBANCES[0]
    object_state = OBJECT_STATES[class_name][0]
    return build_prompt(class_name, scene, viewpoint, lighting,
                        disturbance, object_state)


def get_variation_prompts(class_name: str, count: int = 20,
                          seed: int = 42) -> list[str]:
    """
    Phase B: generate `count` diverse prompts per class.
    Uses deterministic sampling across the variation grid for reproducibility.
    """
    rng = random.Random(seed)
    category = CLASS_CATEGORY[class_name]
    scenes = SCENES[category]
    states = OBJECT_STATES[class_name]

    # Build all possible combinations
    combos = list(product(states, scenes, VIEWPOINTS, LIGHTING, DISTURBANCES))
    rng.shuffle(combos)

    prompts = []
    for state, scene, vp, light, dist in combos[:count]:
        prompts.append(build_prompt(class_name, scene, vp, light, dist, state))

    return prompts


# ─── For testing ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from config import CLASSES

    # Show one baseline prompt
    sample_class = CLASSES[46]  # banana
    print(f"=== Baseline prompt for '{sample_class}' ===")
    print(get_baseline_prompt(sample_class))
    print()

    # Show 3 variation prompts
    print(f"=== 3 variation prompts for '{sample_class}' ===")
    for i, p in enumerate(get_variation_prompts(sample_class, count=3)):
        print(f"\n--- Variation {i+1} ---")
        print(p)

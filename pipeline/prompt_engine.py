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
        "on a camping fold-out table next to a portable stove, tin cans, and a lantern in the woods",
        "on an airplane tray table with a bread roll, a napkin, and the window showing clouds outside",
        "in a hospital room bedside table with a water pitcher, pill bottles, and a TV remote",
        "on a food truck window counter next to other orders, napkins, and condiment dispensers",
        "on a wet balcony table with potted plants, a view of apartment buildings, and raindrops on the surface",
        "on a microwave-adjacent countertop with a kitchen towel, timer display, and scattered seasoning packets",
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
        "in a food court tray with other fast food items, a crumpled receipt, and a plastic fork",
        "on a cooking show countertop with ingredients in bowls, measuring cups, and a recipe card propped up",
        "at an outdoor BBQ table with paper plates, squeeze sauce bottles, corn on the cob, and soda cans everywhere",
        "on a car dashboard next to a coffee cup, sunglasses, and the steering wheel partially visible",
        "inside an open refrigerator shelf with other containers, condiment bottles, and wrapped leftovers",
        "at a food truck window counter with other wrapped items, a chalkboard menu blurry behind, and a hand reaching",
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
        "in a room during moving day with stacked cardboard boxes, bubble wrap on the floor, and a doorway visible",
        "at an outdoor garage sale with price stickers, other furniture pieces, and a folding table of random items",
        "in a sunlit garden patio with outdoor cushions, a charcoal grill nearby, and hedges in the background",
        "in a classroom with other desks, a whiteboard with writing, backpacks on the floor, and buzzing fluorescent lights",
        "in a library reading corner with tall bookshelves, a reading lamp, scattered papers, and a laptop bag",
        "in a child's colorful bedroom with cartoon bedding, stuffed animals, toy bins, and stickers on the furniture",
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
        "a crystal wine glass next to a cheese board with grapes, crackers, and a second glass on a marble counter",
        "a stemless wine glass filled with white wine on a beach blanket with sand, sunscreen, and a paperback book",
        "a wine glass with lipstick marks on the rim sitting on a restaurant bar with cocktail napkins and a menu",
        "three wine glasses of different shapes lined up on a tasting counter with a bottle label visible and notecards",
        "a cracked wine glass base on a kitchen floor with scattered glass shards, a broom, and someone's shoes nearby",
        "a wine glass being held by a hand at arm's length on a rooftop terrace with city lights in the background",
        "an empty wine glass upside down on a dish towel next to drying plates and a sponge near the faucet",
    ],
    "cup": [
        "a coffee cup with steam next to a laptop, phone, and scattered papers on a desk",
        "a tea cup on a saucer beside a teapot, sugar bowl, and a spoon on a tray",
        "a colorful mug on a kitchen counter next to a toaster, banana, and crumby plate",
        "a paper coffee cup on a park bench next to a half-eaten sandwich and a bag",
        "multiple cups of different sizes clustered on a drying rack with plates and bowls",
        "a ceramic cup with a chip on the rim sitting on a bathroom sink next to toothbrushes and soap",
        "a travel mug with a lid on a car console between seats with keys, receipts, and loose coins",
        "a styrofoam cup with a straw on a movie theater armrest with popcorn kernels and a dimly lit screen behind",
        "a handmade pottery cup on a craft fair table with other ceramic pieces, a small price sign, and draped fabric",
        "two mismatched cups on a breakfast table with toast, butter, a jam jar, and a morning newspaper",
        "a plastic cup on its side spilling liquid on a tile floor with a toddler's toys scattered nearby",
        "a cup of hot chocolate with marshmallows on a fireplace mantel next to stockings and holiday candles",
    ],
    "fork": [
        "a fork on a plate with leftover food, next to a knife and a crumpled napkin",
        "a fork with food residue on the tines lying across a bowl of pasta with a glass nearby",
        "two forks and a spoon on a messy table with plates, salt shaker, and bread crumbs",
        "a plastic fork stuck in a takeout container next to another container and a drink cup",
        "a fork partially hidden under a napkin next to a plate of cake and a coffee cup",
        "a silver fork balanced on the edge of a bowl of rice with chopsticks and a soy sauce dish nearby",
        "a fork stuck vertically into a thick piece of steak on a cast iron plate with rosemary and roasted potatoes",
        "a bent fork in a junk drawer with other mixed utensils, a can opener, and tangled rubber bands",
        "a fork on a paper plate at an outdoor BBQ with coleslaw, baked beans, and a checkered tablecloth",
        "a decorative silver fork as part of a formal place setting with a folded cloth napkin and a wine glass",
        "a child's small colorful fork on a high chair tray with mashed food smears, a bib, and a sippy cup",
        "a fork tangled in spaghetti on a plate with garlic bread, a side salad bowl, and a parmesan shaker",
    ],
    "knife": [
        "a butter knife on a plate edge next to toast, a jar of jam, and a cup of coffee",
        "a steak knife with a wooden handle next to a plate of steak with a fork and a glass of wine",
        "a dinner knife next to a folded napkin, a fork, and a bowl of soup on a placemat",
        "a knife on a cutting board with sliced bread, cheese, and a bowl of fruit nearby",
        "a table knife partially under a plate next to scattered food crumbs and a water glass",
        "a bread knife with crumbs on a wooden breadboard next to a loaf of sliced sourdough and a jar of honey",
        "a plastic knife snapped in half on a paper plate with a tough piece of meat and corn on the cob",
        "a cheese knife on a marble serving board with brie, crackers, grapes, and a bottle of wine behind",
        "a dinner knife in a dishwasher rack basket with other silverware, detergent residue, and a filter below",
        "a paring knife on a stained cutting board with lemon halves, fresh herbs, and a bowl of dressing",
        "a decorative knife at a formal place setting with multiple forks, a soup spoon, and crystal glasses",
        "a serrated knife next to a half-cut birthday cake with icing smears, candles, and stacked paper plates",
    ],
    "spoon": [
        "a spoon resting in a bowl of soup next to a plate of bread and a glass of water",
        "a wooden spoon with sauce next to a pot, scattered ingredients, and a messy stovetop",
        "a dessert spoon next to a slice of cake on a plate with a fork and a coffee cup",
        "a spoon sticking out of a yogurt cup next to a banana peel and a granola bar wrapper",
        "multiple spoons and forks jumbled together in a utensil drawer with other kitchen tools",
        "a set of measuring spoons on a flour-dusted countertop with a mixing bowl, eggs, and a recipe book open",
        "a baby spoon on a silicone mat with small baby food jars, a bib, and a highchair tray visible",
        "a large serving spoon in a bubbling pot of chili with steam rising, cornbread on a plate nearby",
        "a teaspoon balanced on a coffee cup edge on a rainy window ledge with raindrops and a notebook below",
        "a tarnished antique spoon on a velvet cloth with other vintage silverware pieces and a jewelry box",
        "a plastic spoon stuck in an ice cream sundae cup with whipped cream, sprinkles, and a cherry on top",
        "a spoon next to a medicine bottle on a nightstand with a digital clock, tissues, and a glass of water",
    ],
    "bowl": [
        "a bowl of salad next to a glass of water, bread basket, and salt and pepper shakers",
        "a chipped porcelain bowl on a counter next to a cutting board, knife, and vegetable scraps",
        "a wooden bowl of mixed fruit on a table with a newspaper, coffee cup, and reading glasses",
        "stacked bowls on a shelf next to plates, cups, and a jar of sugar",
        "a bowl of cereal with a spoon in it next to a milk carton and an orange on a kitchen table",
        "a ramen bowl with chopsticks and a soup ladle on a restaurant counter with a menu and soy sauce bottle",
        "a pet food bowl on a kitchen floor next to a squeaky dog toy, a water bowl, and chair legs",
        "a cracked ceramic bowl with dried pasta inside on a pantry shelf with canned goods and spice jars",
        "a mixing bowl with cake batter and a whisk on a messy baking counter with flour, eggs, and vanilla",
        "a decorative bowl of potpourri on a coffee table with TV remotes, coasters, and a photo album",
        "a bowl of popcorn sitting on a sofa cushion with a fleece blanket, a TV remote, and a dimly lit room",
        "a small dipping bowl on a sushi platter with wasabi, pickled ginger, chopsticks, and edamame pods",
    ],
    "banana": [
        "a ripe banana with brown spots next to an apple, a bowl, and a cup on a kitchen counter",
        "a bunch of bananas in a fruit bowl with oranges and apples, near a knife and cutting board",
        "a peeled banana on a plate next to a bowl of cereal, a spoon, and a glass of juice",
        "a banana partially hidden behind a water bottle on a cluttered office desk with papers",
        "an overripe banana with dark spots on a grocery bag next to bread and canned food",
        "a banana on a school cafeteria table next to an open lunch box, a juice box, and an uneaten sandwich",
        "banana slices on top of a bowl of oatmeal with honey drizzle, blueberries, and a spoon",
        "a banana peel tossed on a park path with a jogger's water bottle and a leashed dog in the distance",
        "a frozen chocolate-dipped banana on a stick with sprinkles, sitting on a parchment-lined baking sheet",
        "a banana zipped into a gym bag partially visible with a towel, headphones, and a protein shaker",
        "a green unripe banana at a farmer's market stall with other tropical fruits and woven baskets",
        "a banana duct-taped to a white gallery wall with a visitor's shoulder visible and a small placard beside it",
    ],
    "apple": [
        "a red apple on a school desk next to books, a pencil case, and a water bottle",
        "a green apple in a fruit bowl with bananas and oranges on a kitchen table near a coffee cup",
        "a half-eaten apple on a messy desk next to a keyboard, phone, and crumpled papers",
        "three apples on a grocery store display partially blocked by a person's shopping basket",
        "an apple next to a sandwich on a cafeteria tray with a milk carton and a napkin",
        "a caramel apple on a stick with chopped nuts on a fall festival table with scattered leaves and small pumpkins",
        "apple slices arranged on a plate with peanut butter dip, celery sticks, and an open lunchbox",
        "a bruised apple in a compost bin with banana peels, coffee grounds, and other food scraps",
        "a single apple among a pile of oranges in a grocery display with a chalkboard price sign",
        "an apple being held up by a child's hand with a playground and other children blurry in the background",
        "a wooden crate of mixed red and green apples at an orchard farm stand with burlap bags and a scale",
        "a sliced apple turning brown on a countertop next to a lemon half, a paring knife, and a cutting board",
    ],
    "sandwich": [
        "a club sandwich on a plate next to a bowl of soup, a glass of soda, and a fork",
        "a grilled cheese with melted cheese oozing on a cutting board next to a knife and chips",
        "a sub sandwich in paper on a restaurant table with condiment bottles, napkins, and a drink cup",
        "a half-eaten sandwich on a park bench next to a coffee cup, bag, and a pigeon nearby",
        "a sandwich on a tray next to fries, ketchup packets, and a crumpled napkin in a fast food place",
        "a sandwich cut diagonally on wax paper inside a brown bag lunch with a juice box and a chips bag",
        "an open-faced sandwich with toppings on a bakery counter with a chalkboard menu and an espresso machine",
        "a panini with dark grill marks on a diner plate with dill pickles, a side of fries, and a ketchup bottle",
        "a sandwich wrapped in foil on a hiking trail rock next to a backpack, water bottle, and a trail map",
        "a BLT sandwich with a toothpick flag on a brunch plate next to a mimosa glass and a fruit salad bowl",
        "a soggy submarine sandwich half-out of a paper bag on a rainy bus stop bench with a closed umbrella",
        "a tiny tea sandwich on a tiered stand at an afternoon tea with scones, jam, and a teapot",
    ],
    "orange": [
        "an orange on a kitchen counter next to a knife, a cutting board, bananas, and a bowl",
        "a halved orange on a plate next to a glass of juice, a spoon, and toast on another plate",
        "oranges in a mesh bag on a grocery cart with other produce items and a shopping list",
        "a peeled orange with segments on a plate next to a napkin and a cup of tea on a coffee table",
        "an orange partially hidden behind a water bottle and a stack of books on a desk",
        "orange segments in a small Tupperware container inside a lunchbox with a thermos, crackers, and a handwritten note",
        "an orange on a sunny windowsill with morning light streaming through, dust motes visible, and curtains framing the view",
        "an orange mid-air being juggled by a person with two other oranges, a kitchen counter in the background",
        "dried orange slices threaded on a string as decoration alongside cinnamon sticks and pinecones on a mantel",
        "an orange with a silly face drawn on it on a child's desk with crayons, construction paper, and a backpack",
        "a blood orange cut open showing vivid red flesh on a dark slate board with a knife and mint leaves",
        "an orange tucked inside a Christmas stocking hanging from a fireplace mantel with ornaments and pine branches",
    ],
    "broccoli": [
        "broccoli florets on a plate next to grilled chicken, rice, and a fork on a dinner table",
        "steamed broccoli in a bowl next to other side dishes, a glass of water, and salt shaker",
        "broccoli on a cutting board with a knife, scattered carrot peels, and other chopped vegetables",
        "roasted broccoli on a baking sheet with other roasted vegetables and tongs",
        "a head of broccoli in a grocery bag next to carrots, onions, and a receipt",
        "broccoli tossed in a sizzling stir-fry wok with bell peppers, tofu cubes, and soy sauce splashes on the stove",
        "a single broccoli tree floret standing upright on a child's plate with other untouched vegetables",
        "broccoli florets in a bamboo steamer basket over a pot with steam, next to a timer and a pot holder",
        "broccoli in a blender jar with spinach, a banana, and protein powder being made into a smoothie on a counter",
        "broccoli growing in a garden bed with rich soil, a small trowel, gardening gloves, and a watering can",
        "overcooked mushy broccoli on a cafeteria tray with mystery meat, a dinner roll, and a carton of milk",
        "broccoli and cheddar soup in a bread bowl on a wooden table with a spoon and crusty bread pieces alongside",
    ],
    "carrot": [
        "carrots with green tops on a cutting board next to a knife, celery, and a bowl of hummus",
        "sliced carrot rounds in a pot of stew with other vegetables and a wooden spoon",
        "a peeled carrot next to a peeler, carrot shavings, and other vegetables on a messy counter",
        "baby carrots in a bowl next to ranch dip, celery sticks, and a plate on a party table",
        "carrots in a grocery bag with other produce, partially spilling onto a kitchen counter",
        "a carrot cake slice with cream cheese frosting on a plate with a dessert fork and a coffee cup",
        "a carrot held out by a small child next to a rabbit in a petting zoo with hay on the ground",
        "shredded carrots scattered on top of a salad bowl with lettuce, cherry tomatoes, croutons, and dressing drizzle",
        "roasted carrots with herbs on a sheet pan next to a roasted whole chicken and golden potatoes",
        "a carrot stick standing in a Bloody Mary cocktail glass with celery on a brunch table among other drinks",
        "carrots in a small school garden with a handwritten plant marker, dark soil, and a child's rain boot visible",
        "pickled carrot sticks in a glass jar on a pantry shelf with other preserved vegetables and mason jars",
    ],
    "hot dog": [
        "a hot dog with mustard in a bun on a paper plate next to fries, a drink cup, and ketchup",
        "two hot dogs on a tray at a ballpark with a beer cup, napkins, and a scoreboard blurry behind",
        "a grilled hot dog on a grill with other sausages, tongs, and smoke rising",
        "a loaded hot dog with toppings next to another hot dog, chips, and a soda can on a table",
        "a hot dog in a wrapper on a street vendor's cart with condiment bottles and other food items",
        "a hot dog topped with sauerkraut on a plate at an Oktoberfest table with beer steins and soft pretzels",
        "a corn dog on a stick with a bite taken out, held by a hand at a state fair with carnival rides behind",
        "mini hot dog appetizers on toothpicks on a party platter next to other finger foods and dipping sauces",
        "a charred hot dog on a campfire skewer next to marshmallows, a glowing log, and sparks in the night air",
        "a hot dog in a bun being assembled by two hands with squeeze bottle toppings lined up on a counter",
        "a Chicago-style hot dog with all the toppings in a red plastic basket with fries and a pickle spear",
        "packaged hot dog buns on a grocery shelf next to relish jars, yellow mustard bottles, and ketchup",
    ],
    "pizza": [
        "a pizza slice on a paper plate next to a soda can, napkins, and other slices in an open box",
        "a whole pizza in a box on a coffee table with a remote control, cups, and a phone",
        "a pizza slice being pulled by a hand with stretchy cheese, other slices and plates around",
        "a pizza on a wooden board in a restaurant with a beer glass, salad bowl, and other tables behind",
        "leftover pizza slices in a greasy box next to crumpled napkins and an empty bottle",
        "a frozen pizza being slid out of its cardboard box on a kitchen counter with an oven preheating behind",
        "a personal-sized pizza on a school cafeteria tray with a side salad cup, a drink, and a napkin",
        "a pizza being sliced by a rotary wheel cutter on a wooden board with flour dust and fresh basil leaves",
        "a cold leftover pizza slice on a plate next to an open fridge with condiment bottles visible inside",
        "a deep dish pizza in a cast iron pan with a serving spatula, on a red checkered tablecloth",
        "a pizza delivery box being opened by eager hands on a doorstep with a welcome mat and shoes visible",
        "a kid's pizza with a smiley face made of vegetable toppings on a colorful plate with crayons and a juice box",
    ],
    "donut": [
        "a glazed donut on a napkin next to a cup of coffee, a phone, and sugar packets on a cafe table",
        "a chocolate donut with a bite taken out on a desk next to a keyboard and a coffee mug",
        "a box of assorted donuts on a break room table with coffee cups, a paper towel roll, and chairs",
        "a powdered sugar donut on a plate with sugar scattered around, next to a glass of milk",
        "a donut in a paper bag next to a coffee cup on a car dashboard with the windshield visible",
        "a donut with rainbow sprinkles on a pink plate at a birthday party table with balloons and streamers",
        "a half-eaten donut on a gym locker room bench next to a water bottle, towel, and workout gloves",
        "donut holes being dipped in chocolate sauce on a fondue plate alongside strawberries and pretzel sticks",
        "a jelly-filled donut with red filling oozing out on a diner counter with a glass coffee pot and menus",
        "freshly fried donuts on a baking tray with a piping bag of glaze, a bowl of sprinkles, and a flour-dusted apron",
        "a stale forgotten donut with a fly on it on an outdoor cafe table with empty cups and crumbs scattered",
        "a mini donut on a paper stick dusted with powdered sugar, held at a fair with carnival lights behind",
    ],
    "cake": [
        "a slice of chocolate cake on a plate with a fork, next to coffee cups and a candle on a table",
        "a birthday cake with messy frosting and partially melted candles, plates and a knife nearby",
        "a cupcake on a desk next to papers, a laptop screen, and a half-empty water bottle",
        "a partially sliced cheesecake on a stand with dessert plates, forks, and crumbs scattered",
        "a slice of cake on a paper plate at an outdoor party with cups, decorations, and chairs around",
        "a tiered wedding cake on a banquet table with flower arrangements, champagne glasses, and place cards",
        "a mug cake in a microwave-safe cup on a dorm room desk with a spoon, an open textbook, and a laptop",
        "a lopsided homemade cake with uneven icing on a counter surrounded by mixing bowls, spatulas, and flour",
        "a cake roll with cream filling sliced on a cutting board with powdered sugar dusting and a serrated knife",
        "a tres leches cake glistening and soaked on a plate with a fork and a tall glass of horchata",
        "a cake on a shelf inside a bakery display case with price tags, other pastries, and glass reflections",
        "a smash cake on a high chair tray with a baby's frosting-covered hands, crumbs everywhere, and balloons behind",
    ],
    "chair": [
        "a wooden dining chair at a table with plates, cups, and a person's bag hanging on the back",
        "an office chair at a cluttered desk with monitors, papers, a coffee cup, and cables",
        "a folding chair on a patio next to a table with drinks, a potted plant, and other chairs",
        "a vintage armchair in a corner with a book on the seat, a lamp beside it, and a rug underneath",
        "multiple chairs around a messy dining table with plates, glasses, and leftover food",
        "a beach chair on sand with a draped towel, flip-flops, an open book, and an umbrella casting a shadow",
        "a broken wooden chair with a missing leg in a cluttered garage with tools, paint cans, and sawdust",
        "a rocking chair on a front porch with a folded blanket, a mug on the armrest, and autumn leaves on the steps",
        "a gaming chair at a desk with RGB LED lights, dual monitors, headphones hanging, and energy drink cans",
        "a stack of white plastic chairs against a wall in a community hall with a folding table and banner posters",
        "a high chair in a kitchen with a baby bib draped over it, scattered cereal pieces, and a sippy cup on the tray",
        "a lawn chair on green grass in a backyard next to a running sprinkler, a kiddie pool, and a garden hose",
    ],
    "couch": [
        "a gray sofa with messy throw pillows, a blanket, a remote, and a phone on the cushions",
        "a leather couch in a living room with a coffee table, books, cups, shoes on the floor, and a TV on",
        "a sectional sofa with a laptop on it, a person's legs visible, pillows on the floor, and a lamp",
        "a loveseat in a small apartment with clothes draped over the arm, a cat, and a side table with clutter",
        "a couch at a waiting room with other chairs, magazines on a table, and scuffed flooring",
        "a floral-patterned sofa in a grandmother's living room with crocheted doilies, porcelain figurines, and a wall clock",
        "a futon folded flat as a couch in a tiny studio apartment with a kitchenette visible, shoes by the door, and a poster",
        "a sofa with a sleeping person's feet sticking out, a pizza box on the coffee table, and the bluish TV glow",
        "an outdoor wicker sofa on a patio with weather-worn cushions, a side table with lemonade, and string lights above",
        "a couch wrapped in moving blankets being loaded onto a truck ramp with a furniture dolly nearby",
        "a leather sofa in a therapist's office corner with a tissue box, a notepad on the armrest, and a framed diploma on the wall",
        "a velvet sofa at a thrift store with a handwritten price tag, other used furniture around, and harsh fluorescent lighting",
    ],
    "potted plant": [
        "a succulent on a windowsill next to other small plants, a watering can, and dusty window",
        "a large monstera plant next to a couch with a coffee table, books, and a cup in the background",
        "a hanging plant with trailing vines above a desk with a monitor, keyboard, and coffee mug",
        "a cactus on a cluttered desk next to a lamp, phone, pens, and sticky notes",
        "multiple potted plants of different sizes on a shelf with books, picture frames, and a clock",
        "a dead potted plant with brown drooping leaves on a neglected balcony with a dusty railing and empty terra cotta pots",
        "a potted herb garden with basil, mint, and rosemary on a kitchen windowsill with spice jars and a recipe card",
        "a large fiddle leaf fig tree in a pot next to a doorway with a coat rack, umbrella stand, and muddy shoes on the mat",
        "an orchid in a decorative ceramic pot on a bathroom counter with toiletries, a steamy mirror, and hanging towels",
        "a potted plant with a get-well ribbon on a hospital room windowsill with greeting cards and a water jug",
        "a tiny succulent planted in a vintage teacup on a vanity table with perfume bottles, jewelry, and a small mirror",
        "a potted fern hanging from a macrame holder in a bohemian living room with woven tapestries and floor cushions",
    ],
    "bed": [
        "a queen bed with rumpled sheets, a phone on the pillow, a nightstand with a lamp and water glass",
        "an unmade bed with clothes thrown on it, shoes on the floor, and a cluttered nightstand",
        "a bed with a laptop on it, pillows against the headboard, a charger cable, and a mug on the nightstand",
        "a twin bed in a dorm room with a desk, chair, books, posters on the wall, and a backpack on the floor",
        "a bed with a cat sleeping on it, a book left open, a glass on the nightstand, and curtains half-drawn",
        "an air mattress on a bare floor with a sleeping bag, a camping lantern, and moving boxes in an otherwise empty room",
        "a hospital bed with side rails up, a call button dangling, an IV stand, and a curtain divider pulled halfway",
        "a bunk bed in a children's room with a ladder, stuffed animals on the top bunk, glow-in-the-dark stars, and a nightlight",
        "a four-poster bed in a historic bedroom with heavy velvet drapes, an antique dresser, and a chandelier above",
        "a dog sprawled sleeping on a human bed with paw prints on the white sheets and a pillow knocked to the floor",
        "a Murphy bed folded out from the wall in a tiny studio apartment with a kitchenette and a narrow writing desk",
        "an outdoor daybed by a turquoise pool with striped cushions, a folded towel, sunglasses, and a tropical drink on a side table",
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

"""
Shared taxonomy for the MemeMatch vision pipeline.

Defines the reaction categories, the CLIP prompt banks used for zero-shot
scoring, the folder-name priors, and the filename keyword priors.
"""

# ---------------------------------------------------------------------------
# Category set.
# The first seven mirror face-api.js expression outputs 1:1 so a detected face
# maps directly onto them. The rest are meme-reaction categories reachable via
# landmark features (smile/mouth/brow), body pose, or blends of the core seven.
# ---------------------------------------------------------------------------
CATEGORIES = [
    "happy", "sad", "angry", "surprised", "fearful", "disgusted", "neutral",
    "laughing", "crying", "smug", "confused", "bored", "love", "awkward",
    "flexing", "pointing", "facepalm", "mocking",
]
CAT_INDEX = {c: i for i, c in enumerate(CATEGORIES)}

# ---------------------------------------------------------------------------
# EXPRESSION prompts: what the face / body in the image literally looks like.
# These carry the most weight because the runtime query is a detected face.
# ---------------------------------------------------------------------------
EXPRESSION_PROMPTS = {
    "happy": [
        "a person with a big happy smile",
        "a smiling face full of joy",
        "a cheerful grinning character",
        "a delighted face beaming with happiness",
        "someone looking genuinely pleased and upbeat",
    ],
    "sad": [
        "a person with a sad depressed face",
        "a sorrowful downcast expression",
        "a gloomy unhappy character looking down",
        "a face full of disappointment and sadness",
        "someone looking miserable and dejected",
    ],
    "angry": [
        "a furious angry face",
        "a person scowling with rage",
        "an enraged character shouting in anger",
        "a face with furrowed brows full of fury",
        "someone looking mad and hostile",
    ],
    "surprised": [
        "a shocked face with wide open eyes",
        "a person gasping in surprise",
        "an astonished character with a dropped jaw",
        "a startled stunned expression",
        "someone looking amazed and caught off guard",
    ],
    "fearful": [
        "a terrified frightened face",
        "a person looking scared and panicked",
        "a character trembling in fear",
        "a horrified anxious expression",
        "someone looking afraid and nervous",
    ],
    "disgusted": [
        "a disgusted grimacing face",
        "a person recoiling with a look of revulsion",
        "a character sneering at something gross",
        "a nauseated repulsed expression",
        "someone looking grossed out and cringing",
    ],
    "neutral": [
        "a blank expressionless face",
        "a person with a flat deadpan stare",
        "an emotionless neutral portrait",
        "a character with no particular expression",
        "someone staring blankly at the camera",
    ],
    "laughing": [
        "a person laughing hysterically",
        "a face crying with laughter",
        "a character howling with laughter, mouth wide open",
        "someone doubled over giggling uncontrollably",
        "an amused face bursting out laughing",
    ],
    "crying": [
        "a person sobbing with tears streaming down",
        "a crying face covered in tears",
        "a character weeping and wiping their eyes",
        "a tearful bawling expression",
        "someone in tears, visibly crying",
    ],
    "smug": [
        "a smug self-satisfied smirk",
        "a person looking arrogant and pleased with themselves",
        "a confident cocky character raising an eyebrow",
        "a proud superior expression",
        "someone giving a knowing condescending smirk",
    ],
    "confused": [
        "a confused puzzled face",
        "a person squinting in bewilderment",
        "a character scratching their head, thinking hard",
        "a perplexed questioning expression",
        "someone looking lost and unsure what is going on",
    ],
    "bored": [
        "a bored unimpressed face",
        "a person rolling their eyes with disinterest",
        "a tired sleepy character zoning out",
        "an apathetic dull expression",
        "someone looking completely uninterested",
    ],
    "love": [
        "a face full of love with heart eyes",
        "a person blushing adoringly",
        "an affectionate character in love",
        "a dreamy infatuated expression",
        "someone gazing lovingly with admiration",
    ],
    "awkward": [
        "an awkward uncomfortable face, sweating nervously",
        "a person with a forced uneasy smile",
        "a character cringing with embarrassment",
        "a nervous anxious sweating expression",
        "someone looking visibly uncomfortable and out of place",
    ],
    "flexing": [
        "a muscular person flexing their biceps",
        "a strong character showing off their muscles",
        "a powerful confident bodybuilder pose",
        "someone striking a strong dominant pose",
        "a buff figure flexing with arms raised",
    ],
    "pointing": [
        "a person pointing their finger at something",
        "a character pointing accusingly at the viewer",
        "someone gesturing and pointing to the side",
        "a figure with an outstretched pointing arm",
        "two people pointing at something together",
    ],
    "facepalm": [
        "a person facepalming with hand on forehead",
        "a character covering their face in disappointment",
        "someone burying their face in their palm",
        "an exasperated figure holding their head in disbelief",
        "a face hidden behind a hand out of shame",
    ],
    "mocking": [
        "a person mocking and taunting someone",
        "a character sticking their tongue out to tease",
        "someone laughing at another person failing",
        "a sarcastic jeering expression",
        "a figure making fun of somebody",
    ],
}

# ---------------------------------------------------------------------------
# VIBE prompts: the overall reaction/meaning of the meme rather than the face.
# Weighted lower than expression, but disambiguates captioned templates.
# ---------------------------------------------------------------------------
VIBE_PROMPTS = {
    "happy": ["a wholesome feel-good meme", "a meme celebrating a win", "a positive uplifting reaction image"],
    "sad": ["a depressing sad meme", "a meme about losing and feeling down", "a melancholic reaction image"],
    "angry": ["an angry rage meme", "a meme expressing outrage", "a hostile confrontational reaction image"],
    "surprised": ["a shocking plot twist meme", "a meme about an unexpected reveal", "a stunned reaction image"],
    "fearful": ["a scary anxiety meme", "a meme about panicking and running away", "a fearful reaction image"],
    "disgusted": ["a cursed disgusting meme", "a meme reacting to something gross", "a cringe reaction image"],
    "neutral": ["a blank meme template with no text", "a plain uncaptioned reaction image", "a neutral deadpan meme"],
    "laughing": ["a hilarious funny meme", "a meme that is extremely funny", "a laughing reaction image"],
    "crying": ["an emotional meme that makes you cry", "a meme about sobbing", "a crying reaction image"],
    "smug": ["a smug superiority meme", "a gigachad meme about being better", "a self-satisfied gloating reaction image"],
    "confused": ["a confusing meme that makes no sense", "a meme about not understanding something", "a puzzled reaction image"],
    "bored": ["a boring uninteresting meme", "a meme about being bored and unimpressed", "an apathetic reaction image"],
    "love": ["a romantic wholesome meme", "a meme about a crush and affection", "a loving reaction image"],
    "awkward": ["an awkward uncomfortable meme", "a meme about a cringeworthy situation", "a nervous sweating reaction image"],
    "flexing": ["a meme about being strong and dominant", "a chad flexing superiority meme", "a powerful bragging reaction image"],
    "pointing": ["a meme where characters point at something", "a meme calling something out", "an accusing reaction image"],
    "facepalm": ["a facepalm disappointment meme", "a meme about a stupid mistake", "an exasperated reaction image"],
    "mocking": ["a mocking spongebob meme", "a meme making fun of someone", "a taunting bullying reaction image"],
}

# ---------------------------------------------------------------------------
# Cat-subject prompt banks.
#
# A CLIP prototype is only as good as its aim. On a library of cat memes, 99%
# of images sit closer to "a photo of a cat" (0.276 mean cosine) than to "a
# photo of a person" (0.200), so prompts written about people spend most of
# their descriptive budget on something that is not in the picture. The
# per-category z-score hides this - a constant offset cannot reorder a column -
# but the separation between categories suffers.
#
# Same 18 categories, same shape. Only the subject changes.
# ---------------------------------------------------------------------------
CAT_EXPRESSION_PROMPTS = {
    "happy": [
        "a happy cat with a big smiling face",
        "a cheerful grinning cat, eyes squinted with joy",
        "a delighted content cat looking pleased",
        "a smiling kitten beaming with happiness",
        "a joyful cat with a wide happy grin",
    ],
    "sad": [
        "a sad cat with droopy sorrowful eyes",
        "a gloomy dejected cat looking down",
        "a miserable unhappy cat with flattened ears",
        "a melancholy cat with a downcast face",
        "a disappointed sad looking kitten",
    ],
    "angry": [
        "an angry hissing cat baring its teeth",
        "a furious cat with flattened ears and a snarl",
        "an enraged cat screaming with its mouth wide open",
        "a hostile growling cat ready to attack",
        "a mad cat glaring with narrowed eyes",
    ],
    "surprised": [
        "a shocked cat with huge wide open eyes",
        "a startled cat with its jaw dropped",
        "an astonished cat staring in disbelief",
        "a stunned cat caught off guard, pupils dilated",
        "a surprised kitten with an open mouth",
    ],
    "fearful": [
        "a terrified cat with dilated pupils",
        "a scared cat puffed up and backing away",
        "a frightened kitten trembling and hiding",
        "a panicked cat with its ears pinned back",
        "an anxious nervous cat looking afraid",
    ],
    "disgusted": [
        "a disgusted cat recoiling from something gross",
        "a cat making a grimacing revolted face",
        "a cat sneering with its lip curled in disgust",
        "a repulsed cat turning away in distaste",
        "a grossed out cat with a scrunched up face",
    ],
    "neutral": [
        "a plain photo of a cat with no particular expression",
        "a cat with a flat deadpan stare",
        "an ordinary expressionless cat portrait",
        "a cat staring blankly at the camera",
        "a calm resting cat with a relaxed face",
    ],
    "laughing": [
        "a laughing cat with its mouth wide open",
        "a cat grinning and giggling with amusement",
        "a cat howling with laughter",
        "an amused cackling cat",
        "a cat mid-laugh with a wide open smiling mouth",
    ],
    "crying": [
        "a crying cat with big teary eyes",
        "a sobbing cat with tears running down its face",
        "a weeping kitten looking heartbroken",
        "a tearful cat about to cry",
        "a cat with watery glistening sad eyes",
    ],
    "smug": [
        "a smug self-satisfied cat with a smirk",
        "an arrogant cat looking pleased with itself",
        "a cocky confident cat with narrowed knowing eyes",
        "a superior looking cat gloating",
        "a proud cat with a condescending expression",
    ],
    "confused": [
        "a confused cat tilting its head",
        "a puzzled cat squinting in bewilderment",
        "a perplexed cat with a question on its face",
        "a baffled kitten that does not understand",
        "a cat looking lost and unsure what is going on",
    ],
    "bored": [
        "a bored unimpressed cat",
        "a sleepy cat zoning out with half closed eyes",
        "an apathetic cat lying around uninterested",
        "a cat looking completely disinterested",
        "a lazy yawning cat, thoroughly unimpressed",
    ],
    "love": [
        "a cat in love with heart eyes",
        "an affectionate cat cuddling and nuzzling",
        "an adoring cat gazing lovingly",
        "a dreamy infatuated kitten",
        "a sweet loving cat being petted",
    ],
    "awkward": [
        "an awkward cat in an uncomfortable situation",
        "a cat with a forced uneasy expression",
        "a nervous cat looking out of place",
        "a cringing embarrassed cat",
        "a cat caught doing something it should not",
    ],
    "flexing": [
        "a muscular buff cat flexing its muscles",
        "a strong swole cat showing off its body",
        "a powerful cat in a bodybuilder pose",
        "a cat standing tall in a dominant pose",
        "a cat with huge arms flexing",
    ],
    "pointing": [
        "a cat pointing at something with its paw",
        "a cat with an outstretched paw gesturing",
        "a cat pointing accusingly at the viewer",
        "a cat reaching out one paw to indicate something",
        "two cats pointing at something together",
    ],
    "facepalm": [
        "a cat covering its face with a paw",
        "a facepalming cat with a paw on its forehead",
        "a cat burying its face in its paws in disappointment",
        "an exasperated cat hiding its eyes behind a paw",
        "a cat with its head in its paws out of shame",
    ],
    "mocking": [
        "a cat mocking and taunting",
        "a cat sticking its tongue out to tease",
        "a smirking cat making fun of someone",
        "a sarcastic jeering cat",
        "a cat laughing at another cat failing",
    ],
}

CAT_VIBE_PROMPTS = {
    "happy": ["a wholesome feel-good cat meme", "a cat meme celebrating a win", "a positive uplifting cat picture"],
    "sad": ["a depressing sad cat meme", "a cat meme about losing and feeling down", "a melancholic cat reaction image"],
    "angry": ["an angry rage cat meme", "a cat meme expressing outrage", "a hostile confrontational cat image"],
    "surprised": ["a shocking cat meme with a twist", "a cat meme about an unexpected reveal", "a stunned cat reaction image"],
    "fearful": ["a scary anxious cat meme", "a cat meme about panicking and running away", "a frightened cat reaction image"],
    "disgusted": ["a cursed disgusting cat meme", "a cat meme reacting to something gross", "a cringe cat image"],
    "neutral": ["a plain uncaptioned cat photo", "an ordinary cat picture with no joke", "a deadpan cat reaction image"],
    "laughing": ["a hilarious funny cat meme", "a cat meme that is extremely funny", "a laughing cat reaction image"],
    "crying": ["an emotional crying cat meme", "a cat meme about sobbing", "a sad crying cat reaction image"],
    "smug": ["a smug superiority cat meme", "a chad cat meme about being better", "a self-satisfied gloating cat image"],
    "confused": ["a confusing cat meme that makes no sense", "a cat meme about not understanding something", "a puzzled cat reaction image"],
    "bored": ["a boring uninteresting cat meme", "a cat meme about being bored and unimpressed", "an apathetic lazy cat image"],
    "love": ["a romantic wholesome cat meme", "a cat meme about affection and cuddles", "a loving cat reaction image"],
    "awkward": ["an awkward uncomfortable cat meme", "a cat meme about a cringeworthy moment", "a nervous cat reaction image"],
    "flexing": ["a cat meme about being strong and dominant", "a buff chad cat meme", "a powerful bragging cat image"],
    "pointing": ["a cat meme where cats point at something", "a cat meme calling something out", "an accusing cat reaction image"],
    "facepalm": ["a facepalm disappointment cat meme", "a cat meme about a stupid mistake", "an exasperated cat reaction image"],
    "mocking": ["a mocking teasing cat meme", "a cat meme making fun of someone", "a taunting cat reaction image"],
}

# subject -> (expression bank, vibe bank). analyze_memes.pick_subject() chooses
# between these by asking the library which subject noun it sits closer to.
SUBJECT_BANKS = {
    "person": (EXPRESSION_PROMPTS, VIBE_PROMPTS),
    "cat": (CAT_EXPRESSION_PROMPTS, CAT_VIBE_PROMPTS),
}


# ---------------------------------------------------------------------------
# Folder priors. The Reactions/ subfolders are human-curated ground truth, but
# several bundle opposite poles ("Funny - Not funny"), so these stay soft and
# let CLIP disambiguate within the folder.
# ---------------------------------------------------------------------------
FOLDER_PRIORS = {
    "Angry - Wicked":                    {"angry": 1.0, "mocking": 0.3, "disgusted": 0.2},
    "Attack - Mockery":                  {"mocking": 1.0, "angry": 0.5, "smug": 0.3},
    "Cursed - NSFW":                     {"disgusted": 0.9, "awkward": 0.5, "surprised": 0.3},
    "Dumb - Genius":                     {"confused": 0.8, "smug": 0.6, "bored": 0.2},
    "Funny - Not funny":                 {"laughing": 1.0, "bored": 0.4, "mocking": 0.3},
    "Horny":                             {"love": 1.0, "awkward": 0.4},
    "Humm - Not interesting - Boring":   {"bored": 1.0, "neutral": 0.5, "confused": 0.2},
    "Liar - Sauce":                      {"smug": 0.7, "confused": 0.5, "mocking": 0.3},
    "No - Stop - Police":                {"angry": 0.7, "disgusted": 0.5, "fearful": 0.3},
    "Offend":                            {"disgusted": 0.8, "angry": 0.6, "mocking": 0.5},
    "Sad - Oof - Lose":                  {"sad": 1.0, "crying": 0.7, "facepalm": 0.2},
    "Sweat - Run away":                  {"fearful": 1.0, "awkward": 0.7},
    "WTF":                               {"surprised": 1.0, "confused": 0.8, "disgusted": 0.2},
    "Yes - Win - Love":                  {"happy": 1.0, "love": 0.5, "smug": 0.4},
}

# ---------------------------------------------------------------------------
# Filename keyword priors. Weakest signal, but filenames here are descriptive
# and often name the exact template, so they are worth a small nudge.
# ---------------------------------------------------------------------------
KEYWORD_PRIORS = {
    "happy":     ["happy", "smile", "smiling", "joy", "celebration", "celebrate", "champagne", "party",
                  "content", "heureux", "brilliant", "wholesome", "yay", "win", "winner", "victory", "nice"],
    "sad":       ["sad", "triste", "depressed", "depression", "lonely", "alone", "heartbreak", "funeral",
                  "funerals", "tombe", "grave", "rip", "lose", "loser", "oof", "regret", "pain", "wojak"],
    "angry":     ["angry", "anger", "mad", "rage", "furious", "fight", "bagarre", "frappe", "strike",
                  "beating", "slap", "punch", "yelling", "scream", "vein", "veine", "tabassage", "attack"],
    "surprised": ["surprised", "surprise", "shocked", "shock", "omg", "whoa", "wow", "unexpected",
                  "unsettled", "disturbed", "twist", "reverse", "gasp"],
    "fearful":   ["scared", "scary", "fear", "terrified", "horror", "peur", "effraye", "creepy",
                  "nightmare", "panic", "sweat", "sweating", "run away", "hiding", "afraid"],
    "disgusted": ["disgusted", "disgust", "gross", "cringe", "ew", "ugly", "hate", "trash", "cursed",
                  "nsfw", "offend", "vomit", "nausea"],
    "neutral":   ["blank", "empty", "template", "textless", "clean", "neutral", "plain", "vector"],
    "laughing":  ["laugh", "laughing", "lol", "funny", "hilarious", "giggle", "rire", "mdr"],
    "crying":    ["crying", "cry", "crie", "pleure", "pleurer", "tears", "tear", "larme", "sob", "weeping"],
    "smug":      ["smug", "chad", "gigachad", "proud", "smirk", "superior", "arrogant", "cocky", "based"],
    "confused":  ["confused", "confusion", "thinking", "pense", "think", "explain", "trying", "question",
                  "doubt", "huh", "hmm", "puzzled", "math", "calcul", "dumb", "stupid"],
    "bored":     ["bored", "boring", "unimpressed", "tired", "sleepy", "yawn", "interesting", "ennui",
                  "humm", "meh", "whatever"],
    "love":      ["love", "kiss", "kissing", "crush", "heart", "amour", "romantic", "horny", "cute",
                  "embrasse", "adorable"],
    "awkward":   ["awkward", "nervous", "uncomfortable", "embarrassed", "sweat", "cringe", "shy", "blush"],
    "flexing":   ["flex", "flexing", "muscle", "buff", "swole", "strong", "power", "dominant", "gym"],
    "pointing":  ["pointing", "point", "doigt", "finger", "showing", "soyjak", "accuse"],
    "facepalm":  ["facepalm", "disappointed", "disappointment", "sigh", "bruh", "shame", "regret"],
    "mocking":   ["mocking", "mockery", "taunt", "tease", "sarcastic", "ironic", "moquerie", "spongebob",
                  "mock", "bully", "ridiculiser"],
}

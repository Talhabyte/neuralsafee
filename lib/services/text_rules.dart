// Rule layer for the NeuralSafe text module.
//
// Why this exists: the text model alone (a) misses very short pleas such as
// "help" or "bachao", (b) ignores negation ("I am not scared of him anymore"),
// (c) reacts to violent words used as jokes or idioms ("killing me with his
// jokes"), and (d) cannot see Roman Urdu spelling variants. These rules run on
// the same text, decide the clear cases deterministically and leave everything
// else to the model. Pure Dart (no Flutter imports) so it is unit testable.
//
// Edit patterns carefully and re-run test/services/text_rules_test.dart after
// any change. Rules cover clear cases only; ambiguous text is left to the model.

class TextRuleVerdict {
  /// Minimum text score (0-100) this rule layer demands. 0 = no rule fired.
  final double floor;

  /// Class to report when the rule layer dominates ('Violence', 'Help', 'Distress', 'Normal').
  final String? cls;

  /// True when an idiom, joke, negation or fiction context was recognised and
  /// no real risk remains in the text.
  final bool benign;
  final String why;

  const TextRuleVerdict({required this.floor, this.cls, this.benign = false, this.why = ''});
}

class _Rule {
  final RegExp re;
  final double floor;
  final String cls;
  _Rule(String src, this.floor, this.cls) : re = RegExp(src);
}

const Map<String, List<String>> _expand = {
  'im': ['i', 'am'],
  'dont': ['do', 'not'],
  'didnt': ['did', 'not'],
  'doesnt': ['does', 'not'],
  'cant': ['cannot'],
  'wont': ['will', 'not'],
  'isnt': ['is', 'not'],
  'wasnt': ['was', 'not'],
  'arent': ['are', 'not'],
  'werent': ['were', 'not'],
  'wouldnt': ['would', 'not'],
  'couldnt': ['could', 'not'],
  'havent': ['have', 'not'],
  'hasnt': ['has', 'not'],
  'hadnt': ['had', 'not'],
  'pls': ['please'],
  'plz': ['please'],
  'plss': ['please'],
};

final Set<String> _firstPerson = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'mujhe', 'mujh', 'mujhko', 'mera', 'meri', 'mere', 'hum', 'humein', 'hamara', 'hamare', 'hamari', 'مجھے', 'مجھ', 'میرا', 'میری', 'میرے', 'ہم', 'ہمیں', 'ہمارا'};
final Set<String> _fiction = {'movie', 'movies', 'film', 'films', 'drama', 'dramas', 'serial', 'show', 'novel', 'book', 'story', 'scene', 'game', 'series', 'episode', 'anime', 'character', 'actor', 'actress', 'cartoon', 'song', 'lyrics', 'dialogue', 'trailer', 'netflix', 'ڈرامہ', 'ڈراما', 'فلم', 'کہانی', 'ناول'};
final Set<String> _saveObj = {'paise', 'paisay', 'paisa', 'waqt', 'time', 'bijli', 'pani', 'petrol', 'energy', 'data', 'battery'};
const Map<String, int> _rank = {'Violence': 3, 'Help': 2, 'Distress': 1, 'Normal': 0};

final List<RegExp> _benign = [
  RegExp(r' ki+l+(?:s|ed|ing|er|ers)? me (?:with|by) (?:his |her |their |the |my |your |these |those |all |such )?(?:jokes?|humou?r|laugh\w*|cuteness|cute|danc\w*|sing\w*|moves|smile|looks|face|silliness|puns|memes|stories|comedy|acting|beauty|style)'),
  RegExp(r' (?:dying|die|dead|dies|died) (?:of|from|with|laughing|laughter)(?: (?:laugh\w*|embarrass\w*|boredom|hunger|curiosity|cold|heat|excitement|shame|cringe|thirst|sleep))?'),
  RegExp(r' (?:dying|die|dies|died) (?:to|for|down|out|hard)'),
  RegExp(r' ki+l+(?:s|ed|ing|er|ers)? (?:this|that|the|my|your|our|an?) (?:exam|test|interview|presentation|game|quiz|paper|workout|race|set|show|pitch|round|audition|viva|assignment)'),
  RegExp(r' ki+l+(?:s|ed|ing|er|ers)? it'),
  RegExp(r' murder\w* (?:you|him|her|them) (?:if|when) (?:you|he|she|they) (?:eat|touch|steal|take|finish|use|break|spill|wake|spoil|drink|play)\w* (?:my|the|our) (?:fries|food|chips|pizza|biscuits|chocolate|snacks|ice cream|cake|lunch|burger|samosa|biryani|chai|tea|coffee|charger|remote|seat|phone charger|dinner|dessert|fries)'),
  RegExp(r' ki+l+(?:s|ed|ing|er|ers)? (?:the )?(?:lights?|engine|music|sound|noise|process|processes|app|apps|tab|tabs|switch|time|power|volume|vibe|mood|task|job|battery|session|server|connection|buzz)'),
  RegExp(r' dead (?:tired|slow|end|battery|serious|sure|silent|quiet|line|link)'),
  RegExp(r' (?:phone|battery|laptop|car|engine|charger|tv|watch|mobile) (?:is |was |got )?(?:dead|died|dying)'),
  RegExp(r' cho+k(?:e|es|ed|ing) (?:on|with) (?:my |a |the |some |his |her )?(?:food|water|bone|bite|tea|coffee|chai|roti|rice|pill|laugh\w*|dust)'),
  RegExp(r' blood (?:test|pressure|sugar|group|donation|bank|report|work)'),
  RegExp(r' (?:hi+t+|hit+s|bea+t|beats|beat) (?:me|us|him|her|them) (?:up|to it|at)\b'),
  RegExp(r' hi+t+ (?:me|us|him|her|them) with (?:a|the) (?:joke|news|idea|text|call|link)'),
  RegExp(r' (?:hi+t+|bea+t) (?:the )?(?:gym|road|sack|books|ball|hay|jackpot|target|button|brakes|traffic|heat|rush|queue|crowd|odds|clock|record|eggs?|batter|cream|dough)'),
  RegExp(r' hit or miss| hit (?:song|movie|show|single)| big hit| (?:heart|drum|dhol|bass) beat'),
  RegExp(r' bloodbath'),
  RegExp(r' (?:football|soccer|cricket|basketball|hockey|rugby|kabaddi|baseball|volleyball|badminton|tennis|squash|futsal|wrestling|karate|judo|gymnastics|swimming|jogging|gym)'),
  RegExp(r' (?:play|plays|played|playing) (?:[^ ]+ ){0,2}(?:match|game|games|sport|sports|tournament|outside|ground|park|field)'),
  RegExp(r' (?:he+l+p+|hlp+|halp|hlep)(?: me| us| him| her| them)? (?:with|to|in) (?:my |the |this |a |these |some |our |his |her |their )?(?:homework|assignment|project|essay|maths?|cooking|code|coding|bags?|luggage|shopping|packing|studies|resume|cv|report|presentation|exam|translation|recipe|cleaning|decorating)'),
  RegExp(r' (?:he+l+p+|hlp+|halp|hlep)(?: me| us)? (?:carry|cook|clean|study|revise|solve|translate|decorate)'),
  RegExp(r' hi+t+ (?:a |the |two |three )?(?:six|sixes|four|fours|boundary|century|wicket|ball|shot|home run|target|jackpot)'),
  RegExp(r' (?:scared|afraid|terrified|frightened|fearful) (?:of|about|for) (?:the )?(?:spiders?|snakes?|ghosts?|heights|dogs?|cats?|dark|horror|insects?|lizards?|cockroach\w*|exams?|tests?|results?|interviews?|failing|failure|flying|water|needles|injections|doctors?|dentist|public speaking|presentations?|math|maths|mice|rats?|clowns?|lifts?|elevators?)'),
  RegExp(r' am afraid (?:i|we|that|it|there|you|this) '),
  RegExp(r' (?:i|we) am (?:not|no longer|never) (?:scared|afraid|terrified|frightened|fearful)(?: of (?:him|her|them|anyone|anything|you))?(?: anymore| any more| now)?'),
  RegExp(r' (?:not|no longer) (?:scared|afraid|terrified|frightened|fearful)(?: of (?:him|her|them|anyone|anything|you))?(?: anymore| any more| now)?'),
  RegExp(r' (?!(?:please|pls|plz|just|why|now|dont|never) )[^ ]+ (?:never|never ever|did not|does not|would never|will never|has never|have never|had never|could never|is not|was not|were not|has not|have not|had not|would not|will not) (?:ever )?(?:hi+t+|hit+s|hits|bea+t|beats|beaten|sla+p+|slaps|touch|touches|abuse|abuses|harm|harms|hurt|hurts|threaten|threatens|kill|kills|raise\w*)(?: me| us| her| him| them| anyone| a [^ ]+)?'),
  RegExp(r' (?:he|she|they) do not (?:hi+t+|hit+s|hits|bea+t|beats|beaten|sla+p+|slaps|touch|touches|abuse|abuses|harm|harms|hurt|hurts|threaten|threatens|kill|kills|raise\w*)(?: me| us| her| him| them| anyone)?'),
  RegExp(r' (?:wo|woh|vo|usne|ye|yeh) (?:[^ ]+ ){0,3}(?:nahi|nahin|nahee|na) (?:[^ ]+ ){0,2}(?:ma+r(?:ta|ti|te|na|ne|a|o|e|do|dega|denge|dalega|dalenge|ega|unga|ungi|di|diya|dia|ke|kar)?|peet\w*|thap+a?r\w*)'),
  RegExp(r' ma+r(?:ta|ti|te|na|ne|a|o|e|do|dega|denge|dalega|dalenge|ega|unga|ungi|di|diya|dia|ke|kar)? (?:[^ ]+ ){0,2}(?:nahi|nahin|nahee)'),
  RegExp(r' (?:mujhe |mujhko |ab |bilkul )*dar+ (?:nahi|nahin|nahee)(?: lag\w*)?'),
  RegExp(r' (?:نہیں|نہ) (?:[^ ]+ ){0,2}(?:مارتا|مارتی|مارتے|مارا)'),
  RegExp(r' ڈر (?:نہیں|نہ)'),
];
final RegExp _pastMarker = RegExp(r'(?:years ago|long ago|a long time ago|in the past|back then|ages ago|when i was (?:a )?(?:child|kid|young|little)|that was over|that is over|no longer|anymore|any more)');
final RegExp _pastAbuse = RegExp(r' (?:used to|once|had) (?:[^ ]+ ){0,3}?(?:hi+t+(?:s|ing)?|bea+t(?:ing|s)?|hurt|sla+p+\w*|abus\w*|threaten\w*|kill\w*)(?: me| her| him| us| them)?');
final RegExp _pleaOnly = RegExp(r'^ (?:(?:please|oh|omg|god|allah|ya|my god) )*(?:(?:he+l+p+|hlp+|halp|hlep)(?: me| us)?|save me|(?:mujhe |mujhko )?(?:ba+cha+o+|ba+cha lo|ba+chalo|ba+chai?(?:ye|yen|iye))(?: mujhe| mujhko)?|madad(?: karo| kijiye| kro| chahiye)?|meri madad (?:karo|kro|kijiye)|sos|call (?:the )?(?:police|ambulance|15|911|1122|999|112)|بچاو+|مجھے بچاو+|مجھے بچا لو|بچا لو|بچالو|مدد|مدد کرو|مدد کریں|میری مدد کرو)(?: (?:please|now|karo|jaldi|abhi))* $');
final RegExp _bachaoToken = RegExp(r'^(?:ba+cha+o+|بچاو+)$');

final List<_Rule> _phraseRules = [
  _Rule(r' ki+l+(?:s|ed|ing|er|ers)? (?:my ?self|myself) ', 96, 'Distress'),
  _Rule(r' (?:end|ending) (?:my life|it all|everything) ', 96, 'Distress'),
  _Rule(r' (?:want|wish|going|gonna|about|wanna|rather|better) (?:to )?(?:die|be dead) ', 92, 'Distress'),
  _Rule(r' (?:suicide|suicidal|khudkushi|khud kushi|خودکشی) ', 96, 'Distress'),
  _Rule(r' (?:gun|knife|weapon|pistol|blade) (?:to|at|on|against|near) (?:me|us|(?:my|her|his|our|the) (?:throat|neck|head|face|back|stomach|body|chest|skin|temple|kids?|children)) ', 92, 'Violence'),
  _Rule(r' (?:has|have|had|got|pulled|took out|brought|showed|holding|holds|pointed|pointing|waved|waving|grabbed|grabs|picked up) (?:a |the |his |her |my )?(?:gun|knife|weapon|pistol|blade) ', 90, 'Violence'),
  _Rule(r' (?:hi+t+(?:s|ing|ed)?|bea+t(?:s|ing|en)?|sla+p+(?:s|ed|ing)?|punch\w*|ki+ck(?:s|ed|ing)?|hurt(?:s|ing)?|push(?:es|ed|ing)?|shov(?:e|es|ed|ing)|cho+k(?:e|es|ed|ing)|burn(?:s|ed|ing|t)|grab(?:s|bed|bing)?|drag(?:s|ged|ging)?|stab(?:s|bed|bing)?|strangl\w*|kill(?:s|ed|ing)?) (?:me|us|her|him|them|my [^ ]+|our [^ ]+|the (?:kids|children|baby|child|girl|woman|boy|man|wife))(?! (?:up|to|at|out)) ', 85, 'Violence'),
  _Rule(r' (?:lock|locks|locked|locking) (?:me|us|her|him|them|the kids|my kids) ', 80, 'Violence'),
  _Rule(r' bleed(?:ing|s) ', 80, 'Violence'),
  _Rule(r' cannot (?:breathe|breath) ', 85, 'Distress'),
  _Rule(r' (?:he+l+p+|hlp+|halp|hlep) (?:me|us)(?!(?: me| us| him| her| them)? (?:with|to|in) (?:my |the |this |a |these |some |our |his |her |their )?(?:homework|assignment|project|essay|maths?|cooking|code|coding|bags?|luggage|shopping|packing|studies|resume|cv|report|presentation|exam|translation|recipe|cleaning|decorating)) ', 85, 'Help'),
  _Rule(r' (?:please|someone|somebody|anyone|anybody|people) (?:please )?(?:he+l+p+|hlp+|halp|hlep)(?!(?: me| us| him| her| them)? (?:with|to|in) (?:my |the |this |a |these |some |our |his |her |their )?(?:homework|assignment|project|essay|maths?|cooking|code|coding|bags?|luggage|shopping|packing|studies|resume|cv|report|presentation|exam|translation|recipe|cleaning|decorating)) ', 85, 'Help'),
  _Rule(r' need (?:your |some |urgent |immediate )?(?:he+l+p+|hlp+|halp|hlep)(?!(?: me| us| him| her| them)? (?:with|to|in) (?:my |the |this |a |these |some |our |his |her |their )?(?:homework|assignment|project|essay|maths?|cooking|code|coding|bags?|luggage|shopping|packing|studies|resume|cv|report|presentation|exam|translation|recipe|cleaning|decorating)) ', 80, 'Help'),
  _Rule(r' call (?:the )?(?:police|ambulance|15|911|1122|999|112) ', 85, 'Help'),
  _Rule(r' call for (?:he+l+p+|hlp+|halp|hlep) ', 85, 'Help'),
  _Rule(r' (?:mujhe|mujhko) (?:ba+cha+o+|ba+cha lo|ba+chalo|ba+chai?(?:ye|yen|iye)) ', 90, 'Help'),
  _Rule(r' (?:مجھے )?بچاو+ ', 90, 'Help'),
  _Rule(r' (?:مجھے )?بچا لو ', 90, 'Help'),
  _Rule(r' (?:takes?|took|taking|snatch\w*|confiscat\w*|controls?|controlling|keeps?|kept|holds?) (?:all |all of |half of )?(?:my|our) (?:salary|pay|paycheck|money|income|wages|earnings|atm|card|cards|phone|passport|id|cnic|documents|savings) ', 62, 'Distress'),
  _Rule(r' (?:will not|does not|did not|never|not|cannot|wont) (?:let|allow|permit) (?:me|us|her) (?:see|meet|talk|speak|call|go|leave|work|study|visit|use|out|have)', 62, 'Distress'),
  _Rule(r' (?:checks?|checking|tracks?|tracking|monitors?|monitoring|reads?|reading|spies|spying) (?:on )?(?:my|me|all my) (?:phone|messages|chats|calls|location|texts|whatsapp)', 60, 'Distress'),
  _Rule(r' (?:nobody|no one) (?:would|will) (?:believe|help|listen)', 62, 'Distress'),
  _Rule(r' (?:isolat\w*|cut me off|keeps me away from) ', 60, 'Distress'),
  _Rule(r' (?:kicked|threw|thrown|throw|throws|kicks|kick) (?:me|us|her|him|them) out ', 65, 'Distress'),
  _Rule(r' (?:ghar se |ghar sa )?nika+l (?:diya|dia|do|dega|denge|dungi|dunga) ', 65, 'Distress'),
  _Rule(r' (?:take|taking|took|takes) (?:my |the |our )?(?:kids|children|child|baby|daughter|son)(?: away)? ', 70, 'Distress'),
  _Rule(r' (?:meri|mera|mere) (?:tankhwah|salary|paise|paisay|phone|mobile|paisa) (?:wo |woh |vo |ye |yeh |usne )?(?:le leta|le leti|chheen|cheen|rakh leta|rakh leti|rakhta|rakhti)', 62, 'Distress'),
  _Rule(r' (?:mujhe|mujhko|mujh ko) (?:ghar se )?(?:bahar|milne|jaane|jane|baat karne|kaam karne) (?:nahi|nahin) (?:deta|dete|deti|diya)', 62, 'Distress'),
  _Rule(r' (?:scared|afraid|terrified|frightened|fearful) (?:of|from) (?:him|her|them|my [^ ]+|the [^ ]+) ', 75, 'Distress'),
  _Rule(r' (?:scared|afraid|terrified) to (?:go|return|come|be|stay|sleep|speak|talk|tell) ', 72, 'Distress'),
  _Rule(r' (?:feel|feeling|feels) (?:unsafe|trapped|helpless|hopeless|threatened) ', 70, 'Distress'),
  _Rule(r' cannot (?:take|do|bear|handle|live|go on)\b', 66, 'Distress'),
  _Rule(r' dar+ lag\w* ', 55, 'Distress'),
  _Rule(r' dar+ (?:raha|rahi|gaya|gayi) ', 55, 'Distress'),
];
final List<_Rule> _tokenRules = [
  _Rule(r'^(?:ki+l+(?:s|ed|ing|er|ers)?)$', 95, 'Violence'),
  _Rule(r'^(?:murder\w*|qatal|قتل)$', 95, 'Violence'),
  _Rule(r'^(?:strangl\w*|suffocat\w*|rap(?:e|es|ed|ing|ist)|molest\w*|assault\w*)$', 95, 'Violence'),
  _Rule(r'^(?:stab(?:s|bed|bing)?)$', 90, 'Violence'),
  _Rule(r'^cho+k(?:e|es|ed|ing)$', 90, 'Violence'),
  _Rule(r'^ma+r(?:ta|ti|te|na|ne|a|o|e|do|dega|denge|dalega|dalenge|ega|unga|ungi|di|diya|dia|ke|kar)?$', 90, 'Violence'),
  _Rule(r'^(?:peet\w*|marpeet|marpit|maarpeet)$', 85, 'Violence'),
  _Rule(r'^(?:thap+a?r|thap+ad|thap+ar\w*|تھپڑ)$', 85, 'Violence'),
  _Rule(r'^(?:مار|مارتا|مارتی|مارتے|مارنا|مارا|مارو|مارے|مارنے|ماردے|ماردیا|ماردو)$', 90, 'Violence'),
  _Rule(r'^(?:chaa?ku|chaqu|chhuri|churi|bandoo?[kq]|pistol|بندوق|چھری|چاقو)$', 80, 'Violence'),
  _Rule(r'^(?:hi+t+(?:s|ing|ed)?|bea+t(?:s|ing|en)?|sla+p+(?:s|ed|ing)?|punch\w*|ki+ck(?:s|ed|ing)?|hurt(?:s|ing)?|push(?:es|ed|ing)?|shov(?:e|es|ed|ing)|cho+k(?:e|es|ed|ing)|burn(?:s|ed|ing|t)|grab(?:s|bed|bing)?|drag(?:s|ged|ging)?|stab(?:s|bed|bing)?|strangl\w*|kill(?:s|ed|ing)?)$', 60, 'Violence'),
  _Rule(r'^(?:gun|guns|knife|knives|weapon|weapons|blade|machete)$', 50, 'Violence'),
  _Rule(r'^(?:threat\w*|threaten\w*|dhamki\w*|dhamkee|دھمکی\w*)$', 72, 'Distress'),
  _Rule(r'^(?:abus\w*|harass\w*|stalk\w*)$', 75, 'Distress'),
  _Rule(r'^(?:hostage)$', 80, 'Distress'),
  _Rule(r'^(?:trapped|unsafe)$', 70, 'Distress'),
  _Rule(r'^(?:scream\w*|shout\w*|yell\w*)$', 58, 'Distress'),
  _Rule(r'^(?:danger\w*|panick?\w*)$', 55, 'Distress'),
  _Rule(r'^(?:scared|afraid|terrified|frightened|fearful)$', 50, 'Distress'),
  _Rule(r'^(?:die|dies|dying|dead)$', 55, 'Distress'),
  _Rule(r'^(?:ڈر|خوف)$', 55, 'Distress'),
  _Rule(r'^(?:madad|مدد)$', 80, 'Help'),
];

class TextRules {
  TextRules._();

  /// Lower-cases, normalises Urdu letters, expands contractions, collapses
  /// stretched letters ("bachaaoooo" -> "bachaaoo") and returns word tokens.
  static List<String> normalize(String text) {
    var t = text.toLowerCase();
    t = t.replaceAll(RegExp(r'[\u2018\u2019\u02bc`\u00b4]'), "'");
    t = t
        .replaceAll(RegExp(r'[\u064a\u0649]'), '\u06cc')
        .replaceAll('\u0643', '\u06a9')
        .replaceAll('\u0647', '\u06c1')
        .replaceAll('\u0624', '\u0648')
        .replaceAll(RegExp(r'[\u0623\u0625]'), '\u0627')
        .replaceAll(RegExp(r'[\u064b-\u065f\u0670\u0640]'), '');
    t = t
        .replaceAll("can't", 'cannot')
        .replaceAll("won't", 'will not')
        .replaceAll("ain't", 'is not')
        .replaceAll("n't", ' not')
        .replaceAll(RegExp(r"'m\b"), ' am')
        .replaceAll(RegExp(r"'re\b"), ' are')
        .replaceAll(RegExp(r"'ll\b"), ' will')
        .replaceAll(RegExp(r"'ve\b"), ' have')
        .replaceAll(RegExp(r"'d\b"), ' would')
        .replaceAll(RegExp(r"'s\b"), '')
        .replaceAll("'", '');
    t = t.replaceAllMapped(
        RegExp(r'(.)\1{2,}', unicode: true), (m) => '${m[1]}${m[1]}');
    final words = RegExp(r'[\p{L}\p{N}]+', unicode: true)
        .allMatches(t)
        .map((m) => m[0]!);
    final tokens = <String>[];
    for (final w in words) {
      final e = _expand[w];
      if (e != null) {
        tokens.addAll(e);
      } else {
        tokens.add(w);
      }
    }
    return tokens;
  }

  static String _pad(Iterable<String> tokens) => ' ${tokens.join(' ')} ';

  static String _repad(String s) =>
      _pad(s.split(' ').where((e) => e.isNotEmpty));

  /// Splits a message into sentences (and 50-word chunks for very long ones)
  /// so one dangerous line cannot be hidden by the model's 128-token limit.
  static List<String> splitSegments(String text) {
    final parts = text
        .split(RegExp(r'[.!?\u061f\u06d4\n\r;]+'))
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty);
    final out = <String>[];
    for (final p in parts) {
      final words = p.split(RegExp(r'\s+'));
      if (words.length <= 60) {
        out.add(p);
      } else {
        for (var i = 0; i < words.length; i += 40) {
          final end = i + 50 > words.length ? words.length : i + 50;
          out.add(words.sublist(i, end).join(' '));
        }
      }
    }
    return out;
  }

  static TextRuleVerdict evaluate(String segment) {
    final tokens = normalize(segment);
    if (tokens.isEmpty) {
      return const TextRuleVerdict(floor: 0, why: 'empty');
    }
    var s = _pad(tokens);

    if (_pleaOnly.hasMatch(s)) {
      return const TextRuleVerdict(floor: 92, cls: 'Help', why: 'plea-only');
    }

    final hasFirst = tokens.any(_firstPerson.contains);
    final hasFiction = tokens.any(_fiction.contains);
    if (hasFiction && !hasFirst) {
      return const TextRuleVerdict(
          floor: 0, cls: 'Normal', benign: true, why: 'fiction');
    }

    var benign = false;
    for (final re in _benign) {
      if (re.hasMatch(s)) {
        benign = true;
        s = _repad(s.replaceAll(re, ' '));
      }
    }
    if (_pastMarker.hasMatch(s) && _pastAbuse.hasMatch(s)) {
      benign = true;
      s = _repad(s.replaceAll(_pastAbuse, ' '));
    }

    final rem = s.split(' ').where((e) => e.isNotEmpty).toList();
    var bestFloor = 0.0;
    String? bestCls;
    var bestWhy = '';
    void consider(double f, String c, String why) {
      if (f > bestFloor ||
          (f == bestFloor && bestCls != null && _rank[c]! > _rank[bestCls!]!)) {
        bestFloor = f;
        bestCls = c;
        bestWhy = why;
      }
    }

    for (final r in _phraseRules) {
      if (r.re.hasMatch(s)) consider(r.floor, r.cls, 'phrase');
    }
    for (var i = 0; i < rem.length; i++) {
      final t = rem[i];
      for (final r in _tokenRules) {
        if (r.re.hasMatch(t)) consider(r.floor, r.cls, 'token:$t');
      }
      if (_bachaoToken.hasMatch(t) && !(i > 0 && _saveObj.contains(rem[i - 1]))) {
        consider(88, 'Help', 'token:$t');
      }
    }

    if (bestFloor == 0) {
      return TextRuleVerdict(
          floor: 0,
          cls: benign ? 'Normal' : null,
          benign: benign,
          why: benign ? 'benign-only' : 'none');
    }
    return TextRuleVerdict(floor: bestFloor, cls: bestCls, why: bestWhy);
  }
}

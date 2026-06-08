# Eval report — `baselines/qwen2.5-3b-instruct`

**Model**: `Qwen/Qwen2.5-3B-Instruct`
**Eval data**: `None`
**Teacher-ceiling mode**: `None`
**Engine / device / dtype**: `hf` / `mps` / `torch.bfloat16`

## Headline

- n = **97**
- correct = **27**
- accuracy = **27.84%**

## Failure-mode buckets

| bucket | count | share |
|---|---:|---:|
| `partial_overlap` | 38 | 39.2% |
| `correct` | 27 | 27.8% |
| `right_action_wrong_inputs` | 18 | 18.6% |
| `dup_count` | 8 | 8.2% |
| `action_wrong` | 6 | 6.2% |

**Within `right_action_wrong_inputs` (input dict mismatch types):**

| kind | count |
|---|---:|
| `value_mismatch` | 9 |
| `missing_default_keys` | 4 |
| `mixed` | 3 |
| `extra_keys` | 2 |

## Step-count breakdown

| group | n | correct | accuracy |
|---|---:|---:|---:|
| single-step | 62 | 15 | 24.2% |
| multi-step  | 35 | 12 | 34.3% |

Golden step-count distribution: `{1: 62, 2: 27, 3: 6, 4: 1, 5: 1}`

## Per-API

| API | n | correct | accuracy |
|---|---:|---:|---:|
| 1Forge | 8 | 4 | 50.0% |
| Abstract Public Holidays | 10 | 4 | 40.0% |
| Lob.com | 8 | 3 | 37.5% |
| Auth0 | 11 | 4 | 36.4% |
| AbuseIPDB | 9 | 3 | 33.3% |
| Axolotl | 13 | 3 | 23.1% |
| A Bíblia Digital | 13 | 3 | 23.1% |
| Améthyste | 10 | 2 | 20.0% |
| Apache Superset | 10 | 1 | 10.0% |
| AniAPI | 5 | 0 | 0.0% |

## Concrete examples (up to 2 per bucket)

### `correct`

**row 0 | api=Axolotl**

- INSTRUCTION: Hey, can you show me a random picture of an axolotl?
- GOLD actions: `['getRandomAxolotlImage']`
- GOLD inputs:  `{}`
- PRED actions: `['getRandomAxolotlImage']`
- PRED inputs:  `{}`
- RESPONSE (first 400c):

```
Thought: The user wants to see a random picture of an axolotl. I will use the getRandomAxolotlImage tool to fetch this image.
Action: getRandomAxolotlImage
Action Input: {}
```

**row 2 | api=Axolotl**

- INSTRUCTION: Can you tell me three interesting facts about axolotls' habitats?
- GOLD actions: `['getAxolotlFacts']`
- GOLD inputs:  `{'category': 'habitat', 'limit': 3}`
- PRED actions: `['getAxolotlFacts']`
- PRED inputs:  `{'category': 'habitat', 'limit': 3}`
- RESPONSE (first 400c):

```
Thought: To find out interesting facts about axolotls' habitats, I need to use the getAxolotlFacts tool. I will specify the category as 'habitat' and limit the number of facts to 3.
Action: getAxolotlFacts
Action Input: {"category": "habitat", "limit": 3}
```

### `right_action_wrong_inputs`

**row 1 | api=Axolotl**

- INSTRUCTION: I'm looking for an axolotl that is wild in color and medium in size. Can you help me find some pictures?
- GOLD actions: `['searchAxolotlImages']`
- GOLD inputs:  `{'color': 'wild', 'gender': '', 'size': 'medium', 'page': 1}`
- PRED actions: `['searchAxolotlImages']`
- PRED inputs:  `{'color': 'wild', 'size': 'medium', 'page': 1}`
- RESPONSE (first 400c):

```
Thought: The user wants to find an axolotl picture that is wild in color and medium in size. I need to use the searchAxolotlImages tool to find these specific axolotl images.
Action: searchAxolotlImages
Action Input: {"color": "wild", "size": "medium", "page": 1}
```

**row 3 | api=Axolotl**

- INSTRUCTION: I need to find out about axolotls' physical characteristics. Can you provide some facts?
- GOLD actions: `['getAxolotlFacts']`
- GOLD inputs:  `{'category': 'physical characteristics', 'limit': 3}`
- PRED actions: `['getAxolotlFacts']`
- PRED inputs:  `{'category': 'physical characteristics', 'limit': 5}`
- RESPONSE (first 400c):

```
Thought: To find out about axolotls' physical characteristics, I need to use the getAxolotlFacts tool and specify the category as 'physical characteristics'. Let's proceed with that.
Action: getAxolotlFacts
Action Input: {"category": "physical characteristics", "limit": 5}
```

### `dup_count`

**row 6 | api=Axolotl**

- INSTRUCTION: My nephew is crazy about axolotls and wants to see what they look like at different stages of growth. Could you find images of small, medium, and large-sized axolotls for him to look at?
- GOLD actions: `['searchAxolotlImages', 'searchAxolotlImages', 'searchAxolotlImages']`
- GOLD inputs:  `{'color': 'wild', 'gender': '', 'size': 'large', 'page': 1}`
- PRED actions: `['searchAxolotlImages']`
- PRED inputs:  `{'color': 'wild', 'gender': 'wild', 'size': ['small', 'medium', 'large'], 'page': 1}`
- RESPONSE (first 400c):

```
Thought: To satisfy the user's request, I need to search for images of axolotls at different stages of growth, specifically small, medium, and large sizes. I will use the searchAxolotlImages tool for this purpose.
Action: searchAxolotlImages
Action Input: {"color": "wild", "gender": "wild", "size": ["small", "medium", "large"], "page": 1}
```

**row 7 | api=Axolotl**

- INSTRUCTION: I saw an axolotl with a unique color the other day, but I can't remember what it was called. Can you find images of axolotls with different colors, especially the more unusual ones? I'd like to figure out which one I saw.
- GOLD actions: `['searchAxolotlImages', 'searchAxolotlImages', 'searchAxolotlImages']`
- GOLD inputs:  `{'color': 'albino', 'gender': '', 'size': '', 'page': 1}`
- PRED actions: `['searchAxolotlImages']`
- PRED inputs:  `{'color': 'wild', 'gender': '', 'size': '', 'page': 1}`
- RESPONSE (first 400c):

```
Thought: The user is looking for images of axolotls with various colors, particularly the more unusual ones. I will use the searchAxolotlImages tool to fetch these images.
Action: searchAxolotlImages
Action Input: {"color": "wild", "gender": "", "size": "", "page": 1}
```

### `partial_overlap`

**row 8 | api=Axolotl**

- INSTRUCTION: I want to share a unique axolotl fact with my coworkers during our virtual lunch break tomorrow. Can you find me a captivating fact about axolotls that will surprise and entertain them? Please include the source URL as well.
- GOLD actions: `['getAxolotlFacts']`
- GOLD inputs:  `{'category': 'habits', 'limit': 1}`
- PRED actions: `['getAxolotlFacts', 'getAxolotlFacts', 'None', 'None']`
- PRED inputs:  `{'category': 'physical characteristics', 'limit': 1}`
- RESPONSE (first 400c):

```
Thought: The user wants an entertaining and surprising fact about axolotls along with its source. I can use the getAxolotlFacts tool to fetch facts and then extract the relevant details.
Action: getAxolotlFacts
Action Input: {"category": "physical characteristics", "limit": 1}
Thought: After fetching the facts, I need to parse the response to extract the fact and its source URL.
Action: getAxolotl
```

**row 10 | api=AniAPI**

- INSTRUCTION: I'm in the mood for a new anime to watch but don't know where to start. Can you find me some mystery and action anime options that were released in 2020 and have a user rating of at least 4 stars?
- GOLD actions: `['searchAnime']`
- GOLD inputs:  `{'filter': "genre = 'Mystery' AND genre = 'Action' AND release_year = 2020 AND user_rating >= 4"}`
- PRED actions: `['searchAnime', 'getRecommendations']`
- PRED inputs:  `{'animeId': '1'}`
- RESPONSE (first 400c):

```
Thought: The user wants recommendations for mystery and action anime released in 2020 with a minimum user rating of 4 stars. To provide this, I need to first search for anime that match these criteria. Afterward, I can fetch more detailed information about them, including their cast and crew, cover images, and brief descriptions.
Action: searchAnime
Action Input: {"query": "mystery action", "filte
```

### `action_wrong`

**row 12 | api=AbuseIPDB**

- INSTRUCTION: I've been dealing with a malicious IP address trying to access my system. Can you help me report this address, 192.168.1.100, and let's categorize it as "brute-force" and "DDoS"? Also, include a comment saying "Multiple failed login attempts and DDoS attacks."
- GOLD actions: `['reportAbusiveIP']`
- GOLD inputs:  `{'ip': '192.168.1.100', 'categories': ['brute-force', 'DDoS'], 'comment': 'Multiple failed login attempts and DDoS attacks.'}`
- PRED actions: `['reportAbuseIP', 'checkIPReputation', 'None', 'None']`
- PRED inputs:  `{'ip': '192.168.1.100', 'categories': ['brute-force', 'DDoS'], 'comment': 'Multiple failed login attempts and DDoS attacks.', 'maxAgeInDays': 30}`
- RESPONSE (first 400c):

```
Thought: To report the malicious IP address, we need to use the AbuseIPDB tool. We will provide the IP address, specify the abuse categories as "brute-force" and "DDoS", and add a comment describing the issue.
Action: reportAbuseIP
Action Input: {"ip": "192.168.1.100", "categories": ["brute-force", "DDoS"], "comment": "Multiple failed login attempts and DDoS attacks."}

Response: Since the respons
```

**row 15 | api=AbuseIPDB**

- INSTRUCTION:  My network is being flooded with spam emails from the IP address 200.0.0.25. I want to report it for spam and phishing. In the comment section, add "Sending multiple spam emails and phishing attempts."

- GOLD actions: `['reportAbusiveIP']`
- GOLD inputs:  `{'ip': '200.0.0.25', 'categories': ['Spam', 'Phishing'], 'comment': 'Sending multiple spam emails and phishing attempts.'}`
- PRED actions: `['reportAbuseIP']`
- PRED inputs:  `{'ip': '200.0.0.25', 'categories': ['spam', 'phishing'], 'comment': 'Sending multiple spam emails and phishing attempts.'}`
- RESPONSE (first 400c):

```
Thought: To report the IP address for spam and phishing activities, we need to use the AbuseIPDB tool. We will provide the IP address and specify the abuse categories as 'spam' and 'phishing'. Additionally, we will include the provided comment in the report.
Action: reportAbuseIP
Action Input: {"ip": "200.0.0.25", "categories": ["spam", "phishing"], "comment": "Sending multiple spam emails and phi
```

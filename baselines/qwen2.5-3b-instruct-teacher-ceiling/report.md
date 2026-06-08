# Eval report — `baselines/qwen2.5-3b-instruct-teacher-ceiling`

**Model**: `Qwen/Qwen2.5-3B-Instruct`
**Eval data**: `None`
**Teacher-ceiling mode**: `True` (demo_source=None)
**Engine / device / dtype**: `hf` / `mps` / `torch.bfloat16`

## Headline

- n = **97**
- correct = **76**
- accuracy = **78.35%**

## Failure-mode buckets

| bucket | count | share |
|---|---:|---:|
| `correct` | 76 | 78.4% |
| `partial_overlap` | 10 | 10.3% |
| `right_action_wrong_inputs` | 5 | 5.2% |
| `dup_count` | 4 | 4.1% |
| `action_wrong` | 2 | 2.1% |

**Within `right_action_wrong_inputs` (input dict mismatch types):**

| kind | count |
|---|---:|
| `missing_default_keys` | 4 |
| `value_mismatch` | 1 |

## Step-count breakdown

| group | n | correct | accuracy |
|---|---:|---:|---:|
| single-step | 62 | 54 | 87.1% |
| multi-step  | 35 | 22 | 62.9% |

Golden step-count distribution: `{1: 62, 2: 27, 3: 6, 4: 1, 5: 1}`

## Per-API

| API | n | correct | accuracy |
|---|---:|---:|---:|
| Lob.com | 8 | 8 | 100.0% |
| Axolotl | 13 | 12 | 92.3% |
| Améthyste | 10 | 9 | 90.0% |
| Abstract Public Holidays | 10 | 9 | 90.0% |
| Apache Superset | 10 | 8 | 80.0% |
| A Bíblia Digital | 13 | 10 | 76.9% |
| Auth0 | 11 | 8 | 72.7% |
| AbuseIPDB | 9 | 6 | 66.7% |
| 1Forge | 8 | 4 | 50.0% |
| AniAPI | 5 | 2 | 40.0% |

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
Thought: The user wants to see a random picture of an axolotl. I need to use the `getRandomAxolotlImage` tool to fetch this image along with its source and description.
Action: getRandomAxolotlImage
Action Input: {}
```

**row 1 | api=Axolotl**

- INSTRUCTION: I'm looking for an axolotl that is wild in color and medium in size. Can you help me find some pictures?
- GOLD actions: `['searchAxolotlImages']`
- GOLD inputs:  `{'color': 'wild', 'gender': '', 'size': 'medium', 'page': 1}`
- PRED actions: `['searchAxolotlImages']`
- PRED inputs:  `{'color': 'wild', 'gender': '', 'size': 'medium', 'page': 1}`
- RESPONSE (first 400c):

```
Thought: To find an axolotl that is wild in color and medium in size, we need to use the searchAxolotlImages tool. We will provide the color as 'wild' and the size as 'medium'. Since no gender is specified, we can leave it empty. Let's proceed with this action.
Action: searchAxolotlImages
Action Input: {"color": "wild", "gender": "", "size": "medium", "page": 1}
```

### `dup_count`

**row 7 | api=Axolotl**

- INSTRUCTION: I saw an axolotl with a unique color the other day, but I can't remember what it was called. Can you find images of axolotls with different colors, especially the more unusual ones? I'd like to figure out which one I saw.
- GOLD actions: `['searchAxolotlImages', 'searchAxolotlImages', 'searchAxolotlImages']`
- GOLD inputs:  `{'color': 'albino', 'gender': '', 'size': '', 'page': 1}`
- PRED actions: `['searchAxolotlImages', 'searchAxolotlImages', 'searchAxolotlImages', 'searchAxolotlImages']`
- PRED inputs:  `{'color': 'other', 'gender': '', 'size': '', 'page': 1}`
- RESPONSE (first 400c):

```
Thought: To find images of axolotls with different colors, especially the more unusual ones, I need to use the searchAxolotlImages tool. I will start by searching for axolotls with wild coloration, then leucistic, and finally albino coloration. This will help me gather a variety of images to compare against the one the user remembers seeing.
Action: searchAxolotlImages
Action Input: {"color": "wil
```

**row 44 | api=1Forge**

- INSTRUCTION: I have 3000 CHF that I want to convert to USD. Can you tell me how much I would get in USD after the conversion? Additionally, I'd like to set a price alert if the CHF/USD exchange rate goes above 1.1.
- GOLD actions: `['convertCurrency', 'setPriceAlert']`
- GOLD inputs:  `{'from': 'CHF', 'to': 'USD', 'amount': 3000, 'currency_pair': 'CHF/USD', 'price': 1.1, 'direction': 'above'}`
- PRED actions: `['convertCurrency', 'setPriceAlert', 'convertCurrency', 'setPriceAlert']`
- PRED inputs:  `{'from': 'CHF', 'to': 'USD', 'amount': 3000, 'currency_pair': 'CHF/USD', 'price': 1.1, 'direction': 'above'}`
- RESPONSE (first 400c):

```
Thought: To answer the user's question, we need to perform two actions. First, we need to convert 3000 Swiss Francs (CHF) into US Dollars (USD). Then, we need to set up a price alert for the CHF/USD exchange rate to notify us if it goes above 1.10.
Action: convertCurrency
Action Input: {"from": "CHF", "to": "USD", "amount": 3000}

Thought: After getting the conversion result, we can proceed to set
```

### `right_action_wrong_inputs`

**row 10 | api=AniAPI**

- INSTRUCTION: I'm in the mood for a new anime to watch but don't know where to start. Can you find me some mystery and action anime options that were released in 2020 and have a user rating of at least 4 stars?
- GOLD actions: `['searchAnime']`
- GOLD inputs:  `{'filter': "genre = 'Mystery' AND genre = 'Action' AND release_year = 2020 AND user_rating >= 4"}`
- PRED actions: `['searchAnime']`
- PRED inputs:  `{}`
- RESPONSE (first 400c):

```
Thought: To find the appropriate anime, I need to use the searchAnime tool and provide it with the required parameters. These include specifying the genres as Mystery and Action, the release year as 2020, and the minimum user rating as 4 stars. Let's proceed with this request.
Action: searchAnime
Action Input: {"query": "", "filters": {"genre": ["Mystery", "Action"], "release_year": 2020, "user_ra
```

**row 26 | api=Auth0**

- INSTRUCTION: I'd like to update my profile with a new profile picture URL, "https://example.com/new_image.jpg". I have the access token "abcd1234" and my user ID is "g-user123." Can you help me make this change?
- GOLD actions: `['updateUserProfile']`
- GOLD inputs:  `{'userId': 'g-user123', 'accessToken': 'abcd1234', 'profileData': {'profilePictureUrl': 'https://example.com/new_image.jpg'}}`
- PRED actions: `['updateUserProfile']`
- PRED inputs:  `{}`
- RESPONSE (first 400c):

```
Thought: To update the user's profile picture, we need to use the updateUserProfile tool. We already have the necessary information: the user ID, the access token, and the new profile picture URL. Let's prepare the Action Input accordingly.
Action: updateUserProfile
Action Input: {"userId": "g-user123", "accessToken": "abcd1234", "profileData": {"profilePictureUrl": "https://example.com/new_image.
```

### `action_wrong`

**row 12 | api=AbuseIPDB**

- INSTRUCTION: I've been dealing with a malicious IP address trying to access my system. Can you help me report this address, 192.168.1.100, and let's categorize it as "brute-force" and "DDoS"? Also, include a comment saying "Multiple failed login attempts and DDoS attacks."
- GOLD actions: `['reportAbusiveIP']`
- GOLD inputs:  `{'ip': '192.168.1.100', 'categories': ['brute-force', 'DDoS'], 'comment': 'Multiple failed login attempts and DDoS attacks.'}`
- PRED actions: `['checkIPReputation']`
- PRED inputs:  `{'ip': '192.168.1.100', 'maxAgeInDays': 30}`
- RESPONSE (first 400c):

```
Thought: To proceed with reporting the malicious IP address, we need to use the `reportAbusiveIP` tool. However, before doing that, I will first check the current reputation of the IP address to ensure it hasn't already been reported under similar categories. This will help us confirm if we need to report it again or not.
Action: checkIPReputation
Action Input: {"ip": "192.168.1.100", "maxAgeInDay
```

**row 48 | api=A Bíblia Digital**

- INSTRUCTION: I want to read the content of chapter 3 in the book of Job from the New International Version. Please retrieve this chapter for me, including the chapter number, book name, version name, and all the verses with their respective verse numbers and text.
- GOLD actions: `['getAllVersesFromChapter']`
- GOLD inputs:  `{'book': 'Job', 'chapter': 3, 'version': 'New International Version'}`
- PRED actions: `['getBookMetadata']`
- PRED inputs:  `{'book': 'Job', 'version': 'New International Version', 'language': 'en'}`
- RESPONSE (first 400c):

```
Thought: To fulfill the user's request, I need to first get the chapter metadata to confirm that it contains chapter 3, then retrieve all the verses in that chapter. Let's start with getting the chapter metadata.
Action: getBookMetadata
Action Input: {"book": "Job", "version": "New International Version", "language": "en"}
```

### `partial_overlap`

**row 17 | api=AbuseIPDB**

- INSTRUCTION: Hey, I've come across an IP address, 168.0.99.1, that I doubt it to be abusive in some aspects. Could you help me check its reputation score? If its reputation score is normal, please report it as "malware" and "hacking"
- GOLD actions: `['checkIPReputation', 'reportAbusiveIP']`
- GOLD inputs:  `{'ip': '168.0.99.1', 'categories': ['malware', 'hacking'], 'comment': 'Suspected malicious activities.'}`
- PRED actions: `['checkIPReputation']`
- PRED inputs:  `{'ip': '168.0.99.1', 'categories': ['malware', 'hacking'], 'comment': 'Suspected malicious activities.'}`
- RESPONSE (first 400c):

```
Thought: First, I need to check the reputation score of the provided IP address to understand if it has any known issues. After that, I will proceed to report it as "malware" and "hacking" if necessary.
Action: checkIPReputation
Action Input: {"ip": "168.0.99.1"}

Thought: Now that I have the reputation score, I can determine if reporting the IP address as "malware" and "hacking" is necessary. If 
```

**row 18 | api=Améthyste**

- INSTRUCTION: I just joined a new Discord server and I want my avatar to stand out. Can you create a unique avatar for me with a "grunge" style in "png" format? My Discord user ID is "123456789"
- GOLD actions: `['generateAvatar']`
- GOLD inputs:  `{'userId': '123456789', 'style': 'grunge', 'format': 'png'}`
- PRED actions: `['searchUser', 'generateAvatar']`
- PRED inputs:  `{'query': '123456789', 'userId': '123456789', 'style': 'grunge', 'format': 'png'}`
- RESPONSE (first 400c):

```
Thought: To create a unique avatar for the user, I need to use the generateAvatar tool. However, I first need to confirm the user ID provided is correct and then proceed with creating the avatar according to the specified style and format.
Action: searchUser
Action Input: {"query": "123456789"}

Thought: After confirming the user ID, I will use the generateAvatar tool to create a grunge-style avat
```

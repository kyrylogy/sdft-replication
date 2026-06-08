# Eval report — `baselines/qwen2.5-3b-instruct-holdout-base`

**Model**: `Qwen/Qwen2.5-3B-Instruct`
**Eval data**: `data/tooluse_data/train_subset_holdout`
**Teacher-ceiling mode**: `False`
**Engine / device / dtype**: `hf` / `mps` / `torch.bfloat16`

## Headline

- n = **100**
- correct = **18**
- accuracy = **18.00%**

## Failure-mode buckets

| bucket | count | share |
|---|---:|---:|
| `partial_overlap` | 46 | 46.0% |
| `correct` | 18 | 18.0% |
| `right_action_wrong_inputs` | 17 | 17.0% |
| `action_wrong` | 10 | 10.0% |
| `dup_count` | 9 | 9.0% |

**Within `right_action_wrong_inputs` (input dict mismatch types):**

| kind | count |
|---|---:|
| `value_mismatch` | 9 |
| `mixed` | 4 |
| `missing_default_keys` | 3 |
| `extra_keys` | 1 |

## Step-count breakdown

| group | n | correct | accuracy |
|---|---:|---:|---:|
| single-step | 63 | 18 | 28.6% |
| multi-step  | 37 | 0 | 0.0% |

Golden step-count distribution: `{1: 63, 2: 19, 3: 10, 4: 6, 6: 1, 8: 1}`

## Per-API

| API | n | correct | accuracy |
|---|---:|---:|---:|
| Open Government, Canada | 1 | 1 | 100.0% |
| Open Government, Estonia | 1 | 1 | 100.0% |
| Excuser | 1 | 1 | 100.0% |
| CurrencyFreaks | 1 | 1 | 100.0% |
| Privacy.com | 1 | 1 | 100.0% |
| Google Safe Browsing | 2 | 2 | 100.0% |
| Digi-Key | 1 | 1 | 100.0% |
| Lichess | 1 | 1 | 100.0% |
| Open Government, Colombia | 1 | 1 | 100.0% |
| Open Government, New South Wales | 1 | 1 | 100.0% |
| Tenders in Romania | 1 | 1 | 100.0% |
| OwlBot | 1 | 1 | 100.0% |
| Imsea | 1 | 1 | 100.0% |
| Geocodify.com | 1 | 1 | 100.0% |
| Lowy Asia Power Index | 1 | 1 | 100.0% |
| apilayer userstack | 1 | 1 | 100.0% |
| Lordicon | 1 | 1 | 100.0% |
| Tronald Dump | 1 | 0 | 0.0% |
| Readme typing SVG | 1 | 0 | 0.0% |
| mod.io | 1 | 0 | 0.0% |
| GeoNames | 1 | 0 | 0.0% |
| Open Skills | 1 | 0 | 0.0% |
| Noctua | 1 | 0 | 0.0% |
| City, Nantes Open Data | 1 | 0 | 0.0% |
| Transport for Atlanta, US | 1 | 0 | 0.0% |
| CryptingUp | 1 | 0 | 0.0% |
| Transport for Ottawa, Canada | 2 | 0 | 0.0% |
| Czech National Bank | 1 | 0 | 0.0% |
| openFDA | 1 | 0 | 0.0% |
| Luchtmeetnet | 2 | 0 | 0.0% |
| Tenders in Hungary | 1 | 0 | 0.0% |
| EPO | 1 | 0 | 0.0% |
| CO2 Offset | 1 | 0 | 0.0% |
| Cutt.ly | 1 | 0 | 0.0% |
| ZipRecruiter | 1 | 0 | 0.0% |
| Pastebin | 1 | 0 | 0.0% |
| Open Government, Greece | 1 | 0 | 0.0% |
| YNAB | 1 | 0 | 0.0% |
| Logs.to | 1 | 0 | 0.0% |
| JCDecaux Bike | 1 | 0 | 0.0% |
| Nutritionix | 1 | 0 | 0.0% |
| Covalent | 2 | 0 | 0.0% |
| Transport for Vancouver, Canada | 1 | 0 | 0.0% |
| Transport for Chicago, US | 1 | 0 | 0.0% |
| Monster Hunter World | 1 | 0 | 0.0% |
| Namedays Calendar | 1 | 0 | 0.0% |
| Zen Quotes | 1 | 0 | 0.0% |
| ZenRows | 1 | 0 | 0.0% |
| Universities List | 1 | 0 | 0.0% |
| Covid-19 Government Response | 1 | 0 | 0.0% |
| TransitLand | 1 | 0 | 0.0% |
| District of Columbia Open Data | 1 | 0 | 0.0% |
| Censys | 2 | 0 | 0.0% |
| Fun Fact | 1 | 0 | 0.0% |
| Shrtcode | 2 | 0 | 0.0% |
| AnimeChan | 1 | 0 | 0.0% |
| geoPlugin | 1 | 0 | 0.0% |
| WakaTime | 1 | 0 | 0.0% |
| Wallhaven | 1 | 0 | 0.0% |
| University of Oslo | 1 | 0 | 0.0% |
| Football-Data | 1 | 0 | 0.0% |
| NewsX | 1 | 0 | 0.0% |
| National Bank of Poland | 1 | 0 | 0.0% |
| GeoDB Cities | 1 | 0 | 0.0% |
| Guild Wars 2 | 1 | 0 | 0.0% |
| Wger | 1 | 0 | 0.0% |
| arXiv | 1 | 0 | 0.0% |
| IUCN | 1 | 0 | 0.0% |
| Metacert | 1 | 0 | 0.0% |
| IFTTT | 1 | 0 | 0.0% |
| JSONbin.io | 1 | 0 | 0.0% |
| Metaphorsum | 1 | 0 | 0.0% |
| Guerrilla Mail | 1 | 0 | 0.0% |
| PoetryDB | 1 | 0 | 0.0% |
| PostalPinCode | 1 | 0 | 0.0% |
| Spanish random names | 1 | 0 | 0.0% |
| Trace Moe | 2 | 0 | 0.0% |
| ELI | 1 | 0 | 0.0% |
| British National Bibliography | 1 | 0 | 0.0% |
| Gaana | 2 | 0 | 0.0% |
| Open Government, Argentina | 1 | 0 | 0.0% |
| MercadoPago | 1 | 0 | 0.0% |
| ipgeolocation | 1 | 0 | 0.0% |
| GoTiny | 1 | 0 | 0.0% |
| MyVaccination | 1 | 0 | 0.0% |
| Line | 1 | 0 | 0.0% |
| LectServe | 1 | 0 | 0.0% |
| WhatJobs | 1 | 0 | 0.0% |
| CleanURI | 1 | 0 | 0.0% |
| Metro Lisboa | 1 | 0 | 0.0% |
| Dogs | 1 | 0 | 0.0% |
| KKBOX | 1 | 0 | 0.0% |

## Concrete examples (up to 2 per bucket)

### `right_action_wrong_inputs`

**row 0 | api=Tronald Dump**

- INSTRUCTION: I'm putting together a collage of Trump's most memorable quotes. Could you show me quotes and tweets having the tag "Election" from the Tronald Dump database?
- GOLD actions: `['getQuotesByTag']`
- GOLD inputs:  `{'tag': 'Election', 'page': 1, 'size': 10}`
- PRED actions: `['getQuotesByTag']`
- PRED inputs:  `{'tag': 'Election'}`
- RESPONSE (first 400c):

```
Thought: To create the collage, we need to fetch quotes and tweets tagged with "Election" from the Tronald Dump database. Let's start by fetching those.
Action: getQuotesByTag
Action Input: {"tag": "Election"}
```

**row 13 | api=Czech National Bank**

- INSTRUCTION: I'm tracking my investments in different currencies, and I need to see how the exchange rates have changed over the past month. Can you provide me with the historical exchange rates for USD, EUR, and GBP between June 1st and June 30th?
- GOLD actions: `['getHistoricalExchangeRates']`
- GOLD inputs:  `{'currencies': ['USD', 'EUR', 'GBP'], 'startDate': '2020-06-01', 'endDate': '2020-06-30'}`
- PRED actions: `['getHistoricalExchangeRates']`
- PRED inputs:  `{'currencies': ['USD', 'EUR', 'GBP'], 'startDate': '2023-06-01', 'endDate': '2023-06-30'}`
- RESPONSE (first 400c):

```
Thought: The user wants to know the historical exchange rates for USD, EUR, and GBP between June 1st and June 30th. To get this information, I need to use the Czech National Bank's service to fetch historical exchange rates.
Action: getHistoricalExchangeRates
Action Input: {"currencies": ["USD", "EUR", "GBP"], "startDate": "2023-06-01", "endDate": "2023-06-30"}
```

### `correct`

**row 1 | api=Open Government, Canada**

- INSTRUCTION: Search for datasets related to education and start the results at index 20.
- GOLD actions: `['searchDatasets']`
- GOLD inputs:  `{'query': 'education', 'limit': 10, 'offset': 20}`
- PRED actions: `['searchDatasets']`
- PRED inputs:  `{'query': 'education', 'limit': 10, 'offset': 20}`
- RESPONSE (first 400c):

```
Thought: To find datasets related to education, I need to use the searchDatasets tool with the appropriate query and offset parameters. Since we want to start the results at index 20, the offset should be set to 20 (as the first result is at index 0).
Action: searchDatasets
Action Input: {"query": "education", "limit": 10, "offset": 20}
```

**row 4 | api=Open Government, Estonia**

- INSTRUCTION: I need detailed information on the dataset with the datasetId "EstPop123". Can you fetch that for me?
- GOLD actions: `['getDatasetDetails']`
- GOLD inputs:  `{'datasetId': 'EstPop123'}`
- PRED actions: `['getDatasetDetails']`
- PRED inputs:  `{'datasetId': 'EstPop123'}`
- RESPONSE (first 400c):

```
Thought: To fetch the detailed information on the dataset with the given datasetId, I need to use the 'getDatasetDetails' tool. Let's proceed with that.
Action: getDatasetDetails
Action Input: {"datasetId": "EstPop123"}
```

### `dup_count`

**row 2 | api=Readme typing SVG**

- INSTRUCTION: I want to create a cool introduction for my personal website. Can you generate a typing SVG with the text "Hi, I'm John Doe" in a 20-point font size, green color, "Arial" font type, typing speed of 120 ms, start delay of 250 ms, deleting speed of 200 ms, and a pause duration of 1.5 seconds for me to embed on my site?
- GOLD actions: `['generateTypingSvg']`
- GOLD inputs:  `{'text': "Hi, I'm John Doe", 'fontSize': 20, 'color': 'green', 'fontType': 'Arial', 'typingSpeed': 120, 'startDelay': 250, 'deletingSpeed': 200, 'pauseDuration': 1500}`
- PRED actions: `['generateTypingSvg', 'generateTypingSvg']`
- PRED inputs:  `{'text': "Hi, I'm John Doe", 'fontSize': 20, 'color': 'green', 'fontType': 'Arial', 'typingSpeed': 120, 'startDelay': 250, 'deletingSpeed': 200, 'pauseDuration': 1500}`
- RESPONSE (first 400c):

```
Thought: The user wants a specific typing SVG with certain parameters. I need to use the 'generateTypingSvg' tool to create this SVG based on the provided details.
Action: generateTypingSvg
Action Input: {"text": "Hi, I'm John Doe", "fontSize": 20, "color": "green", "fontType": "Arial", "typingSpeed": 120, "startDelay": 250, "deletingSpeed": 200, "pauseDuration": 1500}
Thought: After generating th
```

**row 10 | api=CryptingUp**

- INSTRUCTION: I'm quite interested in how the value of Chainlink has developed over the past month. Please display the weekly price, volume, and other indicators for the last four weeks in GBP.
- GOLD actions: `['getHistoricalData', 'getHistoricalData']`
- GOLD inputs:  `{'symbol': 'LINK', 'interval': '1d', 'start': '2020-09-01', 'end': '2020-10-01', 'convert': 'GBP'}`
- PRED actions: `['getHistoricalData']`
- PRED inputs:  `{'symbol': 'LINK', 'interval': '1w', 'start': '2023-03-01', 'end': '2023-04-01', 'convert': 'GBP'}`
- RESPONSE (first 400c):

```
Thought: The user wants to know the development of Chainlink's value over the past month, specifically focusing on the last four weeks. To provide this information, we need to retrieve the historical data for Chainlink with a weekly interval and convert it into GBP. Let's use the 'getHistoricalData' tool to fetch the data.
Action: getHistoricalData
Action Input: {"symbol": "LINK", "interval": "1w"
```

### `partial_overlap`

**row 3 | api=mod.io**

- INSTRUCTION: I need to update some information about my mod. Can you help me with that? I want to change the name and add some tags.
- GOLD actions: `['updateMod']`
- GOLD inputs:  `{'modId': '12345', 'name': 'New Mod Name', 'tags': ['new', 'tags']}`
- PRED actions: `['updateMod', 'getMod']`
- PRED inputs:  `{'modId': 12345, 'name': 'Updated Mod Name', 'summary': 'This is an updated mod with new features.', 'description': 'Detailed description of the updated mod.', 'tags': ['new tag', 'updated tag']}`
- RESPONSE (first 400c):

```
Thought: The user wants to update their mod's information, including changing the name and adding tags. I will use the 'updateMod' tool to accomplish this.
Action: updateMod
Action Input: {"modId": 12345, "name": "Updated Mod Name", "summary": "This is an updated mod with new features.", "description": "Detailed description of the updated mod.", "tags": ["new tag", "updated tag"]}
Thought: After u
```

**row 5 | api=GeoNames**

- INSTRUCTION: Hey, can you retrieve the time zone information for New York City using the GeoNames API?
- GOLD actions: `['getTimeZone']`
- GOLD inputs:  `{'locationId': 5128581}`
- PRED actions: `['getLocationInfo', 'getTimeZone']`
- PRED inputs:  `{'locationName': 'New York City', 'locationId': '<location_id_from_previous_step>'}`
- RESPONSE (first 400c):

```
Thought: To find the time zone information for New York City, I need to first get the location ID for New York City. After that, I can use the location ID to fetch the time zone details.
Action: getLocationInfo
Action Input: {"locationName": "New York City"}
Thought: Now that I have the location ID for New York City, I will use it to fetch the time zone information.
Action: getTimeZone
Action Inpu
```

### `action_wrong`

**row 8 | api=City, Nantes Open Data**

- INSTRUCTION: I'm curious about the traffic flow in a specific area of Nantes. Can you help me obtain the current traffic flow information for that location?
- GOLD actions: `['getLocationTrafficFlow']`
- GOLD inputs:  `{'locationId': '123'}`
- PRED actions: `['None']`
- PRED inputs:  `{}`
- RESPONSE (first 400c):

```
Thought: To provide the user with the current traffic flow information, I need to use the 'getLocationTrafficFlow' tool. However, I first need to find out the location ID of the specific area they're interested in. Since this information isn't provided directly in the question, I'll ask the user to specify the location ID or provide more details so I can proceed.
Action: None
Action Input: null
Re
```

**row 69 | api=arXiv**

- INSTRUCTION: Can you search for recent research papers on quantum computing?
- GOLD actions: `['search']`
- GOLD inputs:  `{'search_query': 'quantum computing', 'start': 0, 'max_results': 10}`
- PRED actions: `['arXiv']`
- PRED inputs:  `{'search_query': 'quantum computing', 'start': 0, 'max_results': 5}`
- RESPONSE (first 400c):

```
Thought: To find recent research papers on quantum computing, I need to use the arXiv search tool with appropriate parameters.
Action: arXiv
Action Input: {"search_query": "quantum computing", "start": 0, "max_results": 5}
```

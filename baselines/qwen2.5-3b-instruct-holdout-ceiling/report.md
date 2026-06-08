# Eval report — `baselines/qwen2.5-3b-instruct-holdout-ceiling`

**Model**: `Qwen/Qwen2.5-3B-Instruct`
**Eval data**: `data/tooluse_data/train_subset_holdout`
**Teacher-ceiling mode**: `True` (demo_source=golden_response)
**Engine / device / dtype**: `hf` / `mps` / `torch.bfloat16`

## Headline

- n = **100**
- correct = **50**
- accuracy = **50.00%**

## Failure-mode buckets

| bucket | count | share |
|---|---:|---:|
| `correct` | 50 | 50.0% |
| `partial_overlap` | 25 | 25.0% |
| `right_action_wrong_inputs` | 14 | 14.0% |
| `dup_count` | 7 | 7.0% |
| `action_wrong` | 3 | 3.0% |
| `no_action` | 1 | 1.0% |

**Within `right_action_wrong_inputs` (input dict mismatch types):**

| kind | count |
|---|---:|
| `value_mismatch` | 10 |
| `missing_default_keys` | 4 |

## Step-count breakdown

| group | n | correct | accuracy |
|---|---:|---:|---:|
| single-step | 63 | 40 | 63.5% |
| multi-step  | 37 | 10 | 27.0% |

Golden step-count distribution: `{1: 63, 2: 19, 3: 10, 4: 6, 6: 1, 8: 1}`

## Per-API

| API | n | correct | accuracy |
|---|---:|---:|---:|
| Tronald Dump | 1 | 1 | 100.0% |
| Readme typing SVG | 1 | 1 | 100.0% |
| Open Government, Estonia | 1 | 1 | 100.0% |
| Noctua | 1 | 1 | 100.0% |
| City, Nantes Open Data | 1 | 1 | 100.0% |
| openFDA | 1 | 1 | 100.0% |
| Tenders in Hungary | 1 | 1 | 100.0% |
| EPO | 1 | 1 | 100.0% |
| CO2 Offset | 1 | 1 | 100.0% |
| Excuser | 1 | 1 | 100.0% |
| CurrencyFreaks | 1 | 1 | 100.0% |
| Open Government, Greece | 1 | 1 | 100.0% |
| Logs.to | 1 | 1 | 100.0% |
| Covalent | 2 | 2 | 100.0% |
| Transport for Chicago, US | 1 | 1 | 100.0% |
| Google Safe Browsing | 2 | 2 | 100.0% |
| Namedays Calendar | 1 | 1 | 100.0% |
| Zen Quotes | 1 | 1 | 100.0% |
| Digi-Key | 1 | 1 | 100.0% |
| Universities List | 1 | 1 | 100.0% |
| Covid-19 Government Response | 1 | 1 | 100.0% |
| Censys | 2 | 2 | 100.0% |
| Open Government, Colombia | 1 | 1 | 100.0% |
| Open Government, New South Wales | 1 | 1 | 100.0% |
| geoPlugin | 1 | 1 | 100.0% |
| Tenders in Romania | 1 | 1 | 100.0% |
| Wallhaven | 1 | 1 | 100.0% |
| OwlBot | 1 | 1 | 100.0% |
| University of Oslo | 1 | 1 | 100.0% |
| NewsX | 1 | 1 | 100.0% |
| Imsea | 1 | 1 | 100.0% |
| Geocodify.com | 1 | 1 | 100.0% |
| arXiv | 1 | 1 | 100.0% |
| IFTTT | 1 | 1 | 100.0% |
| Metaphorsum | 1 | 1 | 100.0% |
| Guerrilla Mail | 1 | 1 | 100.0% |
| PoetryDB | 1 | 1 | 100.0% |
| PostalPinCode | 1 | 1 | 100.0% |
| apilayer userstack | 1 | 1 | 100.0% |
| GoTiny | 1 | 1 | 100.0% |
| Lordicon | 1 | 1 | 100.0% |
| LectServe | 1 | 1 | 100.0% |
| Metro Lisboa | 1 | 1 | 100.0% |
| Transport for Ottawa, Canada | 2 | 1 | 50.0% |
| Luchtmeetnet | 2 | 1 | 50.0% |
| Shrtcode | 2 | 1 | 50.0% |
| Gaana | 2 | 1 | 50.0% |
| Open Government, Canada | 1 | 0 | 0.0% |
| mod.io | 1 | 0 | 0.0% |
| GeoNames | 1 | 0 | 0.0% |
| Open Skills | 1 | 0 | 0.0% |
| Transport for Atlanta, US | 1 | 0 | 0.0% |
| CryptingUp | 1 | 0 | 0.0% |
| Czech National Bank | 1 | 0 | 0.0% |
| Cutt.ly | 1 | 0 | 0.0% |
| ZipRecruiter | 1 | 0 | 0.0% |
| Pastebin | 1 | 0 | 0.0% |
| YNAB | 1 | 0 | 0.0% |
| JCDecaux Bike | 1 | 0 | 0.0% |
| Nutritionix | 1 | 0 | 0.0% |
| Transport for Vancouver, Canada | 1 | 0 | 0.0% |
| Privacy.com | 1 | 0 | 0.0% |
| Monster Hunter World | 1 | 0 | 0.0% |
| ZenRows | 1 | 0 | 0.0% |
| Lichess | 1 | 0 | 0.0% |
| TransitLand | 1 | 0 | 0.0% |
| District of Columbia Open Data | 1 | 0 | 0.0% |
| Fun Fact | 1 | 0 | 0.0% |
| AnimeChan | 1 | 0 | 0.0% |
| WakaTime | 1 | 0 | 0.0% |
| Football-Data | 1 | 0 | 0.0% |
| National Bank of Poland | 1 | 0 | 0.0% |
| GeoDB Cities | 1 | 0 | 0.0% |
| Guild Wars 2 | 1 | 0 | 0.0% |
| Wger | 1 | 0 | 0.0% |
| IUCN | 1 | 0 | 0.0% |
| Metacert | 1 | 0 | 0.0% |
| JSONbin.io | 1 | 0 | 0.0% |
| Spanish random names | 1 | 0 | 0.0% |
| Trace Moe | 2 | 0 | 0.0% |
| ELI | 1 | 0 | 0.0% |
| British National Bibliography | 1 | 0 | 0.0% |
| Lowy Asia Power Index | 1 | 0 | 0.0% |
| Open Government, Argentina | 1 | 0 | 0.0% |
| MercadoPago | 1 | 0 | 0.0% |
| ipgeolocation | 1 | 0 | 0.0% |
| MyVaccination | 1 | 0 | 0.0% |
| Line | 1 | 0 | 0.0% |
| WhatJobs | 1 | 0 | 0.0% |
| CleanURI | 1 | 0 | 0.0% |
| Dogs | 1 | 0 | 0.0% |
| KKBOX | 1 | 0 | 0.0% |

## Concrete examples (up to 2 per bucket)

### `correct`

**row 0 | api=Tronald Dump**

- INSTRUCTION: I'm putting together a collage of Trump's most memorable quotes. Could you show me quotes and tweets having the tag "Election" from the Tronald Dump database?
- GOLD actions: `['getQuotesByTag']`
- GOLD inputs:  `{'tag': 'Election', 'page': 1, 'size': 10}`
- PRED actions: `['getQuotesByTag']`
- PRED inputs:  `{'tag': 'Election', 'page': 1, 'size': 10}`
- RESPONSE (first 400c):

```
Thought: To provide the user with the requested quotes and tweets, I first need to fetch the quotes and tweets tagged with "Election". After retrieving them, I'll present them to the user.
Action: getQuotesByTag
Action Input: {"tag": "Election", "page": 1, "size": 10}
```

**row 2 | api=Readme typing SVG**

- INSTRUCTION: I want to create a cool introduction for my personal website. Can you generate a typing SVG with the text "Hi, I'm John Doe" in a 20-point font size, green color, "Arial" font type, typing speed of 120 ms, start delay of 250 ms, deleting speed of 200 ms, and a pause duration of 1.5 seconds for me to embed on my site?
- GOLD actions: `['generateTypingSvg']`
- GOLD inputs:  `{'text': "Hi, I'm John Doe", 'fontSize': 20, 'color': 'green', 'fontType': 'Arial', 'typingSpeed': 120, 'startDelay': 250, 'deletingSpeed': 200, 'pauseDuration': 1500}`
- PRED actions: `['generateTypingSvg']`
- PRED inputs:  `{'text': "Hi, I'm John Doe", 'fontSize': 20, 'color': 'green', 'fontType': 'Arial', 'typingSpeed': 120, 'startDelay': 250, 'deletingSpeed': 200, 'pauseDuration': 1500}`
- RESPONSE (first 400c):

```
Thought: To generate the desired typing SVG, we need to provide the correct parameters based on the user's request. Let's proceed with calling the generateTypingSvg tool with the provided details.
Action: generateTypingSvg
Action Input: {"text": "Hi, I'm John Doe", "fontSize": 20, "color": "green", "fontType": "Arial", "typingSpeed": 120, "startDelay": 250, "deletingSpeed": 200, "pauseDuration": 1
```

### `right_action_wrong_inputs`

**row 1 | api=Open Government, Canada**

- INSTRUCTION: Search for datasets related to education and start the results at index 20.
- GOLD actions: `['searchDatasets']`
- GOLD inputs:  `{'query': 'education', 'limit': 10, 'offset': 20}`
- PRED actions: `['searchDatasets']`
- PRED inputs:  `{'query': 'education', 'limit': 100, 'offset': 0}`
- RESPONSE (first 400c):

```
Thought: To find datasets related to education, we need to use the searchDatasets tool. However, the question asks us to start the results at index 20, but the current limit parameter only allows us to set the maximum number of results to return. We can achieve this by first finding all the datasets related to education and then filtering them to start from the 20th result. 

Action: searchDataset
```

**row 3 | api=mod.io**

- INSTRUCTION: I need to update some information about my mod. Can you help me with that? I want to change the name and add some tags.
- GOLD actions: `['updateMod']`
- GOLD inputs:  `{'modId': '12345', 'name': 'New Mod Name', 'tags': ['new', 'tags']}`
- PRED actions: `['updateMod']`
- PRED inputs:  `{'modId': '67890', 'name': 'New Mod Name', 'tags': ['new', 'tags']}`
- RESPONSE (first 400c):

```
Thought: To proceed with updating the mod's information, I first need to know the unique identifier (modId) of the mod. Since this information is not provided in the question, I will simulate a scenario where I assume the modId is '67890' for demonstration purposes. After obtaining the modId, I can then use the updateMod tool to make the necessary changes.

Action: updateMod
Action Input: {"modId"
```

### `partial_overlap`

**row 5 | api=GeoNames**

- INSTRUCTION: Hey, can you retrieve the time zone information for New York City using the GeoNames API?
- GOLD actions: `['getTimeZone']`
- GOLD inputs:  `{'locationId': 5128581}`
- PRED actions: `['searchPlaces', 'getTimeZone']`
- PRED inputs:  `{'placeName': 'New York', 'adminDivision': 'NY', 'locationId': 'obtained_location_id'}`
- RESPONSE (first 400c):

```
Thought: To get the time zone information for New York City, we first need to identify its location ID. We can use the GeoNames searchPlaces tool to find the location ID for New York City. After obtaining the location ID, we can then use the getTimeZone tool to retrieve the time zone information.
Action: searchPlaces
Action Input: {"placeName": "New York", "adminDivision": "NY"}

After receiving t
```

**row 9 | api=Transport for Atlanta, US**

- INSTRUCTION: There's a train that I frequently take in the morning, and I'd like to know its current location. Could you provide me with the real-time location data for the train on route #1111, including its latitude, longitude, and timestamp?
- GOLD actions: `['getRealTimeVehicleLocation', 'listLines', 'searchStops', 'getRealTimeVehicleLocation', 'getRealTimeArrivalsAndDepartures', 'getRealTimeVehicleLocation']`
- GOLD inputs:  `{'routeId': 101, 'type': 'train', 'keyword': '1111', 'stopId': 101}`
- PRED actions: `['searchStops']`
- PRED inputs:  `{'keyword': '1111'}`
- RESPONSE (first 400c):

```
Thought: To find the real-time location of the train on route #1111, I first need to identify the correct stop associated with this route. However, the provided tools don't seem to directly provide the stopId for a specific route. Instead, I can try to find the stopId by searching for the stop name that corresponds to route #1111. Once I have the stopId, I can use it to fetch the real-time locatio
```

### `action_wrong`

**row 6 | api=Open Skills**

- INSTRUCTION: I'm considering becoming a data analyst, but I need more information on the skills required for this job. Could you provide me with a list of skills necessary for a data analyst, based on the jobId?
- GOLD actions: `['getSkillsByJob']`
- GOLD inputs:  `{'jobId': 'data_analyst'}`
- PRED actions: `['getJobTitles']`
- PRED inputs:  `{'search': 'data analyst'}`
- RESPONSE (first 400c):

```
Thought: To find out the skills required for a data analyst, I need to first identify the job ID associated with the job title 'data analyst'. Once I have the job ID, I can use it as an input to the getSkillsByJob tool to retrieve the list of skills needed for this role.
Action: getJobTitles
Action Input: {"search": "data analyst"}
```

**row 13 | api=Czech National Bank**

- INSTRUCTION: I'm tracking my investments in different currencies, and I need to see how the exchange rates have changed over the past month. Can you provide me with the historical exchange rates for USD, EUR, and GBP between June 1st and June 30th?
- GOLD actions: `['getHistoricalExchangeRates']`
- GOLD inputs:  `{'currencies': ['USD', 'EUR', 'GBP'], 'startDate': '2020-06-01', 'endDate': '2020-06-30'}`
- PRED actions: `['listAvailableCurrencies']`
- PRED inputs:  `{}`
- RESPONSE (first 400c):

```
Thought: To answer this question, I need to first find out the available currencies that the Czech National Bank supports. After identifying the currencies, I can then proceed to fetch the historical exchange rates for USD, EUR, and GBP between June 1st and June 30th.
Action: listAvailableCurrencies
Action Input: {}
```

### `dup_count`

**row 10 | api=CryptingUp**

- INSTRUCTION: I'm quite interested in how the value of Chainlink has developed over the past month. Please display the weekly price, volume, and other indicators for the last four weeks in GBP.
- GOLD actions: `['getHistoricalData', 'getHistoricalData']`
- GOLD inputs:  `{'symbol': 'LINK', 'interval': '1d', 'start': '2020-09-01', 'end': '2020-10-01', 'convert': 'GBP'}`
- PRED actions: `['getHistoricalData']`
- PRED inputs:  `{'symbol': 'LINK', 'interval': '1d', 'start': '2020-09-01', 'end': '2020-09-07', 'convert': 'GBP'}`
- RESPONSE (first 400c):

```
Thought: It seems I need to fetch daily historical data for Chainlink in GBP for the past four weeks. However, the provided tool only allows specifying a single date range at a time. To achieve the goal of showing the value development over the past month, I will need to request data for each week separately and then compile the results. Let me first gather the data for the first week (September 1
```

**row 20 | api=Cutt.ly**

- INSTRUCTION: Someone sent me a shortened link, but I'd like to know the original URL before clicking on it. Can you expand this link for me: "https://cutt.ly/abc123"?
- GOLD actions: `['expandLink']`
- GOLD inputs:  `{'apiKey': '123456789', 'shortUrl': 'https://cutt.ly/abc123'}`
- PRED actions: `['expandLink', 'expandLink']`
- PRED inputs:  `{'apiKey': 'mockAPIKey', 'shortUrl': 'https://cutt.ly/abc123'}`
- RESPONSE (first 400c):

```
Thought: To expand the shortened link, I need to use the expandLink tool. However, I don't have the API key provided in the request. Normally, I would need the API key to proceed. Since we don't have that information, let's assume I can still perform the action and provide a mock response.
Action: expandLink
Action Input: {"apiKey": "mockAPIKey", "shortUrl": "https://cutt.ly/abc123"}

Now answer w
```

### `no_action`

**row 71 | api=Metacert**

- INSTRUCTION: Could you verify if this link leads to a legitimate website?
- GOLD actions: `['verifyLink']`
- GOLD inputs:  `{'link': 'www.example.com'}`
- PRED actions: `[]`
- PRED inputs:  `{}`
- RESPONSE (first 400c):

```
To verify if the provided link leads to a legitimate website, I will use the verifyLink tool. However, please provide me with the actual link that needs to be verified.
```

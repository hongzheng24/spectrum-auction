# CS1440/2440 Final Project: Spectrum Auction

## Introduction

The Spectrum Auction (LSVM18): This final project is your chance to put everything we've learned into practice, especially the cool strategies from the last few labs. It's going to be remote and a lot more free-form than what we're used to, so there's plenty of room to experiment and find what works best for your agent. I hope you all have fun!

## Setup and Installation

Follow these steps to set up your environment and install the necessary package for the lab.

### Step 1: Git Clone the Repository

Open your terminal and navigate to where you want to clone the repository

```bash
git clone https://github.com/brown-agt/Spectrum-Auction-Stencil.git
```

### Step 2: Create a Virtual Environment

Please then navigate to your project directory. Run the following commands to create a Python virtual environment named `.venv`.

If you own a Mac

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

If you own a Windows

```bash
python3.11 -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install the agt server package

```bash
pip install --upgrade pip
pip install --upgrade agt-server
```

## Agent Methods

For the `LSVM Agent`s here are a few methods that you may find helpful!

### Basic Information Methods

- **`get_regional_good()`**:
  - Returns the specific good designated as the regional good for this agent, crucial for regional bidders to focus their strategies around.

- **`get_goods()`**:
  - Retrieves a set of all the goods' names available in the auction, allowing the agent to understand the full scope of what's being auctioned.

- **`is_national_bidder()`**:
  - Checks if the agent is classified as a national bidder, a key distinction that affects bidding strategy and interest in goods.

- **`get_shape()`**:
  - Provides the dimensions of the goods space as a tuple, indicating how goods are organized and potentially their proximity to each other.

- **`get_num_goods()`**:
  - Returns the total number of goods available in the auction, giving the agent a sense of the auction's scale.

- **`get_goods_to_index()`**:
  - Retrieves a dictionary mapping goods names to their indices (tuples), facilitating the conversion between good names and their index locations in valuation arrays.

- **`get_tentative_allocation()`**:
  - Returns a set of strings representing goods tentatively allocated to this agent, showing the current potential gains from the auction.
  
- **`get_current_round()`**:
  - Provides the current round number in the auction, allowing the agent to adjust strategies over time.

- **`get_goods_in_proximity()`**:
  - Lists names of goods within the agent's regional proximity or all goods if a national bidder, guiding regional bidding strategies.

- **`proximity(arr, regional_good)`**:
  - Filters goods based on their proximity to a specified regional good, relevant for regional bidders to evaluate their interest in nearby goods.

#### Utility Calculation Methods

- **`calc_total_valuation(bundle)`**:
  - Calculates the total valuation of a given bundle of goods for the agent

- **`calc_total_prices(bundle)`**:
  - Calculates the total prices of a given bundle of goods for the agent

- **`calc_total_utility(bundle)`**:
  - Calculates the total utility of a given bundle of goods for the agent

#### Valuation and Bidding Methods

- **`get_valuation_as_array()`**:
  - Returns the agent's valuations as a numpy array, offering a structured view of the agent's value assessment of all goods.

- **`get_valuation(good)`**, **`get_valuations(bundle)`**:
  - Retrieves the valuation for specific goods or a set, aiding in determining how much to bid for particular items.

- **`get_min_bids_as_array()`**, **`get_min_bids(bundle)`**:
  - Provides the minimum bids required either as a numpy array (taking in indices) or a map taking in good names. For `get_min_bids` if bundle is not provided it just returns the map for all goods, otherwise a bundle or set of goods can be provided for which you want the min_bids.

- **`is_valid_bid_bundle(my_bids)`**:
  - Checks if a set of bids is valid according to game rules and agent's constraints, preventing invalid bid submissions.

- **`clip_bids(my_bids)`**, **`clip_bid(good, bid)`**:
  - Adjusts bids to meet or exceed the minimum bids, aligning bids with auction rules for either a map of goods to bids, or for a single good and bid.

#### Conversion Methods

- **`map_to_ndarray(map, object)`**, **`ndarray_to_map(arr)`**:
  - Converts between mappings of goods to values and numpy arrays, useful for data manipulation and analysis. This is useful as there are a lot of helper functions that either return `_as_array()` or as a map. This will allow you to easily convert between the two.

### History Methods

Detailed history methods allow the agent to analyze its performance and adjust strategies accordingly. These are returned in a `game_report` that contains a dictionary of the history of the auction that the agent has access to. Both `game_report` and the class has access to the following methods to retrieve different parts of the `game_report`. [E.g. `agent.get_game_report().get_util_history() == agent.get_util_history()`]

- **`get_game_report()`**: Retrieves the Game Report object containing historical information that the agent has access to.

- **`get_util_history()`**: Retrieves a list of the agent's utility values over time, showing how well the agent has been performing.
- **`get_bid_history()`** and **`get_bid_history_map()`**: Offers a history of the bids the agent has made, either as numpy arrays or mappings.
- **`get_price_history()`** and **`get_price_history_map()`**: Gives a record of the prices over the course of the auction.
- **`get_winner_history()`** and **`get_winner_history_map()`**: Details the history of who won what bids in previous rounds, helping the agent to adjust its future bids based on past success rates.

- **`get_previous_util()`**: Provides the most recent utility value for the agent.
- **`get_previous_bid_history()`** and **`get_previous_bid_history_map()`**: Provides the most recent bids the agent has made, either as numpy arrays or mappings.
- **`get_current_prices()`** and **`get_current_prices_map()`**: Provides the most recent prices in the auction.
- **`get_previous_winner_history()`** and **`get_previous_winner_history_map()`**: Provides the most recent history of who won what bids in the last round.

## Notes about the python code

- Please refer to the final project handout and read through it carefully! It contains a lot of information specific to this implementation of the spectrum auction and making sure your code works for submission.
- In class, I mentioned that you need to do relative imports but that is outdated news, the autograder should automatically handle that for you now so as long as you don't name your files random.py or numpy.py or [Insert Common Package Name].py, the submission should work.
- Please let us know if you want to import any new packages not natively provided in agt_server and we will install it on our end after checking it so that the code can import it in the final submission. [E.g. last time we needed to actually install tensorflow for the tensorflow code to run correctly]





Tournament Results
-----------------
Final util 0

Extended Results: 
              Agent 1        Agent 2        Agent 3      Agent 4          Agent 5          Agent 6      A1 Score      A2 Score   A3 Score    A4 Score    A5 Score      A6 Score
    0   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder  Truthful Bidder  1.465006e+01  2.868866e+01  22.460500   45.302710   52.783430  2.805650e+01
    1   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder           keyreg  3.081620e+00  9.262905e+01  21.146209   47.340274   70.586910  3.587073e+01
    2   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder  Truthful Bidder           keyreg -1.368912e-02  8.564216e+00   0.000000  158.467895  143.411020  3.546141e-01
    3   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent   Min Bidder  Truthful Bidder           keyreg  2.761002e+01  3.491046e+00  23.125426  121.765368   71.689528  1.534697e+01
    4   CP - MyAgent  CP2 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  7.370859e+00 -1.828258e+17  47.735961   41.779539   29.161224 -6.729737e+13
    5   CP - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg -1.116362e+18 -6.443751e+09  43.484635   25.282761   15.484217 -3.684571e+11
    6  CP2 - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  2.373959e+01  4.543837e+01  21.910917   40.950336   14.013268  7.551075e+00
Results: 
            Agent Name   Final Score
    2      Jump Bidder  7.408863e+01
    4  Truthful Bidder  5.826276e+01
    3       Min Bidder  5.215318e+01
    1    CP3 - MyAgent -1.516177e+09
    5           keyreg -1.592137e+13
    6    CP2 - MyAgent -4.301783e+16
    0     CP - MyAgent -2.232724e+17
    246.07814383506775 Seconds Elapsed



=================
Less than mid bid bug fixed Results
=================
    Extended Results: 
          Agent 1        Agent 2        Agent 3      Agent 4          Agent 5          Agent 6   A1 Score      A2 Score   A3 Score   A4 Score   A5 Score      A6 Score
0   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder  Truthful Bidder   3.804899  9.044654e+01   2.425550  15.604047  18.897609  1.116711e+01
1   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder           keyreg   1.010913  1.014700e+02  13.174063   9.457420  17.393433  2.278622e+01
2   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder  Truthful Bidder           keyreg   0.256193  1.544026e+02   1.223435  45.418729  33.912936  5.005562e+00
3   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent   Min Bidder  Truthful Bidder           keyreg  19.569206  1.166011e+02   1.917636  36.219156   7.796155  3.070650e+00
4   CP - MyAgent  CP2 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg   5.548555  6.907165e+01  16.050134  17.706610   8.763783  4.853379e+00
5   CP - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  -0.241482 -3.988758e+17  47.818711  50.477764  29.198404 -1.882287e+08
6  CP2 - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  92.518292  7.099916e+00   3.009646  33.201806   8.296646  1.896748e+00
Results: 
         Agent Name   Final Score
6    CP2 - MyAgent  1.143152e+02
3       Min Bidder  3.168015e+01
2      Jump Bidder  3.054149e+01
4  Truthful Bidder  1.983518e+01
0     CP - MyAgent  5.156275e+00
5           keyreg -4.182859e+07
1    CP3 - MyAgent -8.397386e+16




========================
Normalized feature results
========================

Min Bidder won Good G at Price 12.01, Good A at Price 8.10, Good M at Price 8.10, Good H at Price 12.39.                                                                                                                                          
Min Bidder got a final utility of 57.1693424455798
CP - MyAgent won Good I at Price 13.88, Good O at Price 11.13, Good J at Price 16.56, Good K at Price 14.83, Good B at Price 12.72, Good P at Price 9.59, Good C at Price 17.19, Good D at Price 12.43, Good N at Price 11.18.
CP - MyAgent got a final utility of 149.69896427783505
Jump Bidder won Good E at Price 12.38, Good Q at Price 12.53, Good F at Price 15.41.
Jump Bidder got a final utility of 9.119206694047506
CP2 - MyAgent got a final utility of 0
keyreg won Good R at Price 3.50, Good L at Price 4.96.
keyreg got a final utility of 21.74844099541866
Truthful Bidder got a final utility of 0
File saved to Auction 126: Min Bidder VS CP - MyAgent VS Jump Bidder VS CP2 - MyAgent VS keyreg VS Truthful Bidder (National).json.gz                                                                                                             
Extended Results: 
          Agent 1        Agent 2        Agent 3      Agent 4          Agent 5          Agent 6    A1 Score      A2 Score   A3 Score   A4 Score   A5 Score      A6 Score
0   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder  Truthful Bidder   86.766518  7.439565e+00  -0.222645  20.271170  42.147012  1.404669e+01
1   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder           keyreg   94.611391  1.016282e+01   3.320368   9.282956  25.577444  2.166278e+00
2   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder  Truthful Bidder           keyreg  160.294948  2.081310e+00   1.091293  62.158333  37.680645  1.622290e+00
3   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent   Min Bidder  Truthful Bidder           keyreg   96.768400  6.976254e+00   1.225758  37.661117  14.612595  2.539284e+00
4   CP - MyAgent  CP2 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  118.579213 -2.120765e+00   7.748555  26.054673  12.500920  2.803485e+00
5   CP - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg   87.303446  5.388107e+00  28.155435   8.414609   6.747516  7.787945e+00
6  CP2 - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg    7.369968 -8.294243e+08  61.882994  56.079329  27.814032 -1.324774e+19
Results: 
         Agent Name   Final Score
6     CP - MyAgent  1.035159e+02
3       Min Bidder  3.610475e+01
2      Jump Bidder  3.606320e+01
4  Truthful Bidder  1.896138e+01
0    CP2 - MyAgent  5.962238e+00
1    CP3 - MyAgent -2.182696e+08
5           keyreg -3.679928e+18
651.6059987545013 Seconds Elapsed


=========
Aggressive bidding results
========

Min Bidder won Good F at Price 0.10, Good Q at Price 0.10, Good A at Price 0.00, Good E at Price 0.10, Good L at Price 0.10, Good C at Price 0.10, Good K at Price 0.10, Good J at Price 0.10, Good D at Price 0.10.                              
Min Bidder got a final utility of 187.14884774140145
CP3 - MyAgent got a final utility of 0
keyreg won Good R at Price 0.00.
keyreg got a final utility of 0.0
CP2 - MyAgent got a final utility of 0
CP - MyAgent got a final utility of 0
Truthful Bidder won Good P at Price 0.10, Good B at Price 0.10, Good G at Price 0.10, Good M at Price 0.10, Good I at Price 0.10, Good N at Price 0.10, Good H at Price 0.10, Good O at Price 0.10.
Truthful Bidder got a final utility of 224.78624961387413
File saved to Auction 125: Min Bidder VS CP3 - MyAgent VS keyreg VS CP2 - MyAgent VS CP - MyAgent (National) VS Truthful Bidder.json.gz                                                                                                           
Auction 126: Min Bidder VS CP3 - MyAgent VS keyreg VS CP2 - MyAgent VS CP - MyAgent VS Truthful Bidder (National)
Running Round:   2%|███                                                                                                                                                                                         | 17/1053 [00:00<00:41, 24.82it/s]keyreg timed out after 1 seconds.
Checkpoint saved to checkpoints/dqn_model.pt
Checkpoint saved to checkpoints/dqn_model.pt
Running Round:   2%|████▋                                                                                                                                                                                       | 26/1053 [00:03<03:12,  5.34it/s]Checkpoint saved to checkpoints/dqn_model.pt
Running Round:   4%|███████▋                                                                                                                                                                                    | 43/1053 [00:05<01:08, 14.68it/s]Checkpoint saved to checkpoints/dqn_model.pt
Min Bidder won Good H at Price 6.60, Good B at Price 6.30, Good G at Price 6.60, Good M at Price 6.60.                                                                                                                                            
Min Bidder got a final utility of 63.43946528286579
CP3 - MyAgent got a final utility of 0
keyreg timed out 1 times
keyreg got a final utility of 0
CP2 - MyAgent got a final utility of 0
CP - MyAgent won Good P at Price 6.25, Good Q at Price 5.95, Good E at Price 3.90, Good I at Price 4.40, Good L at Price 4.00, Good O at Price 4.50, Good J at Price 6.08, Good D at Price 4.50.
CP - MyAgent got a final utility of 147.95598561106794
Truthful Bidder won Good A at Price 6.60, Good R at Price 6.60, Good N at Price 6.60, Good C at Price 6.60, Good K at Price 4.10, Good F at Price 6.60.
Truthful Bidder got a final utility of 7.260377469159494
File saved to Auction 126: Min Bidder VS CP3 - MyAgent VS keyreg VS CP2 - MyAgent VS CP - MyAgent VS Truthful Bidder (National).json.gz                                                                                                           
Extended Results: 
          Agent 1        Agent 2        Agent 3      Agent 4          Agent 5          Agent 6      A1 Score   A2 Score      A3 Score   A4 Score   A5 Score      A6 Score
0   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder  Truthful Bidder  3.989842e+00   3.219476  5.847495e+01  27.482215  50.449634  3.039218e+01
1   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder           keyreg  4.497282e+00  11.919370  2.788431e+01  55.388463  46.942753  8.578667e+00
2   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder  Truthful Bidder           keyreg -2.677531e+19   7.556224 -1.309537e+16  50.489777  62.433026 -8.506723e+18
3   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent   Min Bidder  Truthful Bidder           keyreg  1.127279e+01  13.203459  6.363668e+01  31.909889  24.101245  1.536140e+01
4   CP - MyAgent  CP2 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  6.332581e+01  31.978575  3.399681e+01  30.910967   7.247476  4.461952e+00
5   CP - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  1.300968e+01  24.341954  1.794232e+01  39.717974  31.064604  9.391038e+00
6  CP2 - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder           keyreg  2.438558e+01  31.592021  3.090701e+01  37.019890  27.193960  4.785463e+00
Results: 
         Agent Name   Final Score
6       Min Bidder  3.896480e+01
3      Jump Bidder  3.772711e+01
4  Truthful Bidder  3.405810e+01
1    CP2 - MyAgent  1.353192e+01
2    CP3 - MyAgent -2.619074e+15
5           keyreg -1.790889e+18
0     CP - MyAgent -5.950069e+18
579.1732757091522 Seconds Elapsed




=========
Regularization Run 2
========
Min Bidder won Good N at Price 8.00, Good O at Price 8.00.                                                                                                                                                                                        
Min Bidder got a final utility of 25.109859396804115
Jump Bidder won Good G at Price 13.16, Good C at Price 14.94, Good A at Price 7.26, Good M at Price 13.44.
Jump Bidder got a final utility of 28.53298235345335
keyreg won Good L at Price 8.00, Good K at Price 3.49, Good P at Price 6.00, Good F at Price 3.72, Good R at Price 4.10, Good Q at Price 3.50.
keyreg timed out 1 times
keyreg got a final utility of 168.0745603791632
CP3 - MyAgent won Good H at Price 9.65.
CP3 - MyAgent timed out 1 times
CP3 - MyAgent got a final utility of 6.8573942595411435
CP2 - MyAgent won Good B at Price 7.93, Good I at Price 3.52.
CP2 - MyAgent got a final utility of 12.808475514127341
Truthful Bidder won Good D at Price 8.26, Good J at Price 6.74, Good E at Price 6.12.
Truthful Bidder got a final utility of 0.27033245492557256
File saved to Auction 126: Min Bidder VS Jump Bidder VS keyreg VS CP3 - MyAgent VS CP2 - MyAgent VS Truthful Bidder (National).json.gz                                                                                                            
Extended Results: 
          Agent 1        Agent 2        Agent 3      Agent 4          Agent 5 Agent 6      A1 Score      A2 Score      A3 Score    A4 Score    A5 Score      A6 Score
0   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder       Min Bidder  keyreg -3.435529e+13 -1.508951e+17 -6.132488e+12   27.504473   31.098648 -1.847536e+14
1   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent  Jump Bidder  Truthful Bidder  keyreg  0.000000e+00  0.000000e+00  1.010139e+01  147.848003  179.554046  0.000000e+00
2   CP - MyAgent  CP2 - MyAgent  CP3 - MyAgent   Min Bidder  Truthful Bidder  keyreg  4.471524e+01  2.273264e+01  3.068740e+00   70.074899   58.486637  1.267770e+01
3   CP - MyAgent  CP2 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder  keyreg  5.642861e+01  6.586286e+00  2.670460e+01   37.142988   20.768884  7.483045e+00
4   CP - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder  keyreg  1.332688e+01  1.725397e+01  2.415225e+01   51.474650   46.491323  1.114210e+01
5  CP2 - MyAgent  CP3 - MyAgent    Jump Bidder   Min Bidder  Truthful Bidder  keyreg  6.923106e+00  1.875420e+01  3.053917e+01   67.987265   41.083173  1.145521e+01
Results: 
         Agent Name   Final Score
4       Min Bidder  4.713155e+01
6  Truthful Bidder  4.115348e+01
3      Jump Bidder  3.374547e+01
2    CP3 - MyAgent -1.533122e+12
0     CP - MyAgent -6.062699e+12
5           keyreg -2.639337e+13
1    CP2 - MyAgent -2.382554e+16
781.7207450866699 Seconds Elapsed
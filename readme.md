These are the datasets and implementation for the paper:
The lateest code is available at https://github.com/THUIR/Brain-Relevance-Feedback.

***Relevance Feedback with Brain Signals. Ziyi Ye, Xiaohui Xie, Qingyao Ai, Yiqun Liu, Zhihong Wang, and Min Zhang, 2023.*** submitted to ACM TOIS.

For further information on the proposed relevance feedback method and experimental results, please refer to the paper. 

If you have any problem about this work or dataset, please contact with Ziyi Ye at Yeziyi1998@gmail.com or Yiqun Liu at yiqunliu@tsinghua.edu.cn. We are very willing to help you with the usage of our public data and code and we really hope the related communities (i.e., BMI for search) can attract more attentions.

## Dataset
Please download the user study dataset from https://cloud.tsinghua.edu.cn/d/afe41a6a30bb4a4bbca2/ and save it as "release/".
The dataset description can be find at "release/readme.md".

## Source Codes
All implementation codes are available in the "src" directory.
In order to follow the paper, we suggest to run the codes from source file with prefix "part0" to source file with prefix "part6".
Before running the codes, please check these issues: (1) install all requirements in "requirements.txt" (e.g., using "pip install -r requirements"); (2) download the user study dataset and unzip it as "release/".; (3) create a directory namely "results/" to save experimental results.

### Codes for the brain decoding experiments

|  **Source file**  | **Function**  |
|   :----   |   :----   |
| part0_generate_eeg_score.py | Generate the results of the brain decoding experiment and saved at the directory "release/idx2eeg_score/". |
| part1_data_analyses.py | Analyze and show the brain decoding performance, plot each participant's brain decoding performance with matplotlib. |
| part2_topo_significance.py | Plot the Topography which shows the significance of difference (F-value) between brain response to relevant/irrelevant Web pages. Highlighted channels indicate the differences are significant at p-value<1e−3. |
|system/| Base code for the user study system, being used by source files running brain decoding experiments. |

### Codes for the online testing analyses
|  **Source file**  | **Function**  |
|   :----   |   :----   |
| part1_online_glm.py | The general linear model (glm) analyses for the IRF task during online testing. |
| part1_online_pairwise.py | The pair-wise comparasion for the RRF task during online testing. |


### Codes for IRF and RRF experiments
|  **Source file**  | **Function**  |
|   :----   |   :----   |
| part3_bert_functions.py & part3_bm25.py | Base models for our proposed RF models, being used by source files running IRF and RRF experiments. |
| part3_udr.py | Generate IRF results for given combination parameters. |
| part3_rdr.py | Generate RRF results for given combination parameters. |
| part3_generate_all_udr.py | Generate IRF results for all combination parameters. |
| part3_generate_all_rdr.py | Generate RRF results for all combination parameters. |
| part3_search_fixed_parameter.py | Tune the fixed parameter for IRF and RRF, respectively. |


### Codes for exploring the best combination parameters
|  **Source file**  | **Function**  |
|   :----   |   :----   |
| part4_generate_udr_best_para.py | Search and generate the best combination parameters with synthetic signals in IRF task. Note that the combination parameter is generated for each participants. |
| part4_generate_rdr_best_para.py | Search and generate the best combination parameters with synthetic signals in RRF task. Note that the combination parameter is generated for each participants. |
| part4_generate_best_para2.py | Save the best combination parameters generated with synthetic signals, run this script after running "part4_generate_udr_best_para.py" or "part4_generate_rdr_best_para.py".|


### Codes for indepth analyses and plotting
|  **Source file**  | **Function**  |
|   :----   |   :----   |
| part5_context_analyses_utils.py | Base functions for search context-aware in-depth analyses, being used by source files "part5_context_analyses_paint.py" and "part5_context_analyses_statistics.py". |
| part5_context_analyses_statistics.py | Print the RF performance and in-depth analyses results for the paper. |
| part5_context_analyses_paint.py | Plot the RF performance and in-depth analyses results for the paper with matplotib. |


### Codes for adaptive signals combination
|  **Source file**  | **Function**  |
|   :----   |   :----   |
| part6_simulating_utils.py | Base functions for search and generated the best combination parameters with synthetic signals.  |
| part6_simulating_udr.py | Search and generate the best combination parameters with synthetic signals in IRF task. Note that the combination parameter is generated for each possible search context. |
| part6_simulating_rdr.py | Search and generate the best combination parameters with synthetic signals in RRF task. Note that the combination parameter is generated for each possible search context. |
| part6_udr.py | Run the IRF experiment, similar to "part3_udr.py", but with adaptive signals combination. |
| part6_rdr.py | Run the RRF experiment, similar to "part3_rdr.py", but with adaptive signals combination. |

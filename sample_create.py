import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

# txt2matrix
def txt2matrix(filename):
    file = open(filename)
    lines = file.readlines()
    rows = len(lines)
    datamat = np.zeros((rows, 3))
    row = 0
    for line in lines:
        line = line.split(' ')
        x = line[0:2]
        datamat[row, :] = line[:]
        row = row + 1

    return datamat


# # 获取观测样本o属于状态s的概率
# def prob_O2S(model, o):
#     M_O2S = model["M_O2S"]
#     return M_O2S[:,int(o)]
#
#
# # 根据概率分布生产一个样本
# def get_one_sample_from_Prob_distribution(Prob_dis):
#     N_segment = np.shape(Prob_dis)[0]
#     prob_segment = np.zeros(N_segment)
#
#     for i in range(N_segment):
#         prob_segment[i] = prob_segment[i-1] + Prob_dis[i]
#
#     S = 0
#
#     data = np.random.rand()
#     for i in range(N_segment):
#         if data <= prob_segment[i]:
#             S = i
#             break
#
#     return S
#
#
# # 生成样本
# def get_sample_from_HMM(model, N):
#     M_O2S = model["M_O2S"]
#     datas = np.zeros(N)
#     stats = np.zeros(N)
#
#     # 初始化，根据初始状态生成第一个样本
#     init_S = get_one_sample_from_Prob_distribution(model["pi"])
#     stats[0] = init_S
#     datas[0] = get_one_sample_from_Prob_distribution(M_O2S[int(stats[0])])
#
#     # 生产其它样本
#     for i in range(1, N):
#         # 根据前一个状态，生成当前状态，例：前一个为status1，下一个状态的概率分布即为A[1]
#         stats[i] = get_one_sample_from_Prob_distribution(model["A"][int(stats[i-1])])
#         # 根据当前状态生产一个数据，例：状态1对应概率分布为：M_O2S[1]
#         datas[i] = get_one_sample_from_Prob_distribution(M_O2S[int(stats[i])])
#     return datas, stats
#
#
# # 前向算法 alpha = [N_sample, N_stats]
# # 表示已知前t-1个样本的情况下，第t个样本属于状态i的概率
# def calc_alpha(model, observations):
#     o = observations
#     N_samples = np.shape(o)[0]
#     N_stats = np.shape(model["pi"])[0]
#
#     # alpha初始化
#     alpha = np.zeros([N_samples, N_stats])
#
#     alpha[0] = model["pi"]*model["B"](model, o[0])
#
#     # 计算第0个样本属于第i个状态的概率
#     for t in range(1, N_samples):
#         s_current = np.dot(alpha[t-1], model["A"])
#         alpha[t] = s_current * model["B"](model, o[t])
#
#     return alpha
#
#
# def forward(model, observation):
#     o = observation
#
#     # 前向概率计算
#     alpha = calc_alpha(model, o)
#     prob_seq_f = np.sum(alpha[-1])
#
#     return np.log(prob_seq_f)
#
#
# # 后向算法
# def calc_beta(model, observation):
#     o = observation
#     N_sample = np.shape(o)[0]
#     N_stats = np.shape(model["pi"])[0]
#
#     # beta初始化
#     beta = np.zeros([N_sample, N_stats])
#     # 反向初始值
#     beta[-1] = 1
#
#     # 由t+1时刻的β值及t+1时刻的观测值计算t+1时刻的状态值
#     for t in range(N_sample-2, -1, -1):
#         s_next = beta[t+1] * model["B"](model, o[t+1])
#         beta[t] = np.dot(s_next, model["A"].T)
#
#     return beta
#
# def backward(model, observation):
#     o = observation
#
#     # 计算后向概率
#     beta = calc_beta(model, o)
#     s_next = beta[0] * model["B"](model, o[0])
#     prob_seq_b = np.dot(s_next, model["pi"])
#
#     return np.log(prob_seq_b)


def classification_xyv(states, datas_test):
    states = np.array(list(states[1]))
    states = states[:, np.newaxis]
    datas_states = np.hstack((datas_test, states))

    df = pd.DataFrame(datas_states, columns=['x', 'y', 'Velocity', 'Classification results'])
    df.to_csv('classification_results_of_subpaths.csv', encoding="utf_8_sig")


def classification_v(states, datas):
    states = np.array(list(states[1]))
    states = states[:, np.newaxis]
    datas_states = np.hstack((datas, states))

    df = pd.DataFrame(datas_states, columns=['Velocity', 'Classification results'])
    df.to_csv('classification_results.csv', encoding="utf_8_sig")


def classsification_result(datas, result):
    a = []
    for key in result:
        a = a + result[key]
    a = np.array(a)
    a = a[:, np.newaxis]
    datas_states = np.hstack((datas, a))
    df = pd.DataFrame(datas_states, columns=['x', 'y', 'Velocity', 'Classification results'])
    df.to_csv('ternary_classification_results.csv', encoding="utf_8_sig")



def classification_round_1(datas, n_compoments, states):
    classification_round_1 = list(states[1])
    first_index_of_states = np.zeros(n_compoments+1)
    for i in range(0, n_compoments):
        first_index_of_states[i] = classification_round_1.index(i)

    length = len(classification_round_1)
    first_index_of_states[n_compoments] = length
    first_index_of_states.sort()
    states_info = {}
    for i in range(0, n_compoments):
        filename = "section" + str(i) + ".txt"
        start = (int)(first_index_of_states[i])
        end = (int)(first_index_of_states[i+1])
        section = datas[start:end, :]
        states_info[i] = section
        df = pd.DataFrame(section)
        df.to_csv(filename)

    return states_info


def section_classification(datas):
        m_GMMHMM1 = GMMHMM(n_components=3, n_mix=2, covariance_type='full', n_iter=50, tol=0.00001, verbose=True)
        m_GMMHMM1.fit(datas)
        x = m_GMMHMM1.decode(datas)
        # x[0]:score x[1]:results
        return x[1]

def classification_round_2(states_info, mode):
    '''
    :param states_info: Eye movement trajectory after the first round of classification
    :param mode: mode 0 represents classify based on v,mode 1 represents classify based on x,y,v
    :return: list of eye movement types after second round of classification
    '''
    gaze_state_info = {}
    if mode == 0:
        for key in states_info:
            temp = states_info[key]
            temp = temp[:, 2]
            temp = temp.reshape(-1, 1)
            gaze_state_info[key] = section_classification(temp)
    elif mode == 1:
        for key in states_info:
            gaze_state_info[key] = section_classification(states_info[key])

    return gaze_state_info
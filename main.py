import dataplot
from hmmlearn.hmm import MultinomialHMM, GMMHMM
import sample_create


if __name__ == '__main__':
    ''' task 0 load data '''
    datas= sample_create.txt2matrix('collect_clasi_train.txt')
    datas = datas[:, 0:3]
    # datas = datas.reshape(-1, 1)
    # print(datas)

    ''' task 1 init'''
    # Number of stimulated fixation points
    n_components = 3
    m_GMMHMM = GMMHMM(n_components=n_components, n_mix=2, covariance_type='full', n_iter=50, tol=0.00001, verbose=True)
    m_GMMHMM.fit(datas)

    '''Step 1: divide the whole eye movement sequence into n sub-sequences; 
       Step 2: Extract fixation points in each eye movement path'''
    dataplot.gaze_trace_xyv_plot(datas)
    states = m_GMMHMM.decode(datas)
    # print(states)
    # Results of the first round of classification
    states_info = sample_create.classification_round_1(datas, n_components, states)
    # print(states_info)
    dataplot.path_classification_plot(states_info)

    # Second round of classification
    gaze_states_info = sample_create.classification_round_2(states_info, 0)
    print(gaze_states_info)
    dataplot.gaze_point_classification_plot(states_info, gaze_states_info, 2)
    result = dataplot.gazepoint_classification_final_results(states_info, gaze_states_info)
    sample_create.classsification_result(datas, result)
    print("-----------------stop----------------------")
    # sample_create.classification_xyv(states, datas)

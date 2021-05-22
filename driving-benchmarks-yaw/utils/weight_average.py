'''

Weight를 미리 계산해서 적어놓음
인덱스 1 ~ N 까지에 대해
밑이 0.5인 log를 이용해
=LOG(G16/10, 0.5)
exp를 계산하고
=EXP(H28)

softmax를 적용해서 합이 1인 weight 값으로 만듬
'''


class predefined_weights():

    def getWeight(self, queue_size):
        if queue_size == 1:
            return [1]
        elif queue_size == 2:
            return [0.731058579 * 2, 0.268941421 * 2]
        elif queue_size == 3:
            return [0.635794633 * 3, 0.233895774 * 3, 0.130309593 * 3]
        elif queue_size == 4:
            return [0.585421756 * 4, 0.215364628 * 4, 0.119985396 * 4, 0.079228219 * 4]
        elif queue_size == 5:
            return [0.553631842 * 5, 0.203669773 * 5, 0.113469879 * 5, 0.074925922 * 5, 0.054302585 * 5]
        elif queue_size == 6:
            return [0.531447503 * 6, 0.195508611 * 6, 0.108923077 * 6, 0.071923598 * 6, 0.05212665 * 6, 0.040070561 * 6]
        elif queue_size == 7:
            return [0.514928326 * 7, 0.189431545 * 7, 0.105537382 * 7, 0.069687971 * 7, 0.050506378 * 7, 0.038825033 * 7, 0.031083365 * 7]
        elif queue_size == 8:
            return [0.502057201 * 8, 0.184696522 * 8, 0.102899374 * 8, 0.067946053 * 8, 0.049243923 * 8, 0.037854564 * 8, 0.030306407 * 8, 0.024995956 * 8]
        elif queue_size == 9:
            return [0.491687612 * 9, 0.180881764 * 9, 0.10077407 * 9, 0.066542682 * 9, 0.048226829 * 9, 0.037072708 * 9, 0.029680452 * 9, 0.024479685 * 9, 0.020654198 * 9]
        elif queue_size == 10:
            return [0.483116285 * 10, 0.177728549 * 10, 0.099017329 * 10, 0.065382679 * 10, 0.047386117 * 10, 0.03642644 * 10, 0.029163049 * 10, 0.024052944 * 10, 0.020294144 * 10, 0.017432378 * 10]
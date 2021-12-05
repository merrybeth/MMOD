import numpy as np
import simpy as sp
import statistics as st
import matplotlib.pyplot as plt


def get_theoretical_final_probabilities(l, m1, m2):
    l_1, l_2, l_4 = (l + m1 + m2), (l + m2), (l + m1)
    mm, mm2, mm3, m2m2, m3m2, m2m3 = m1 * m2, m1 * (m2 ** 2), m1 * (m2 ** 3), (m1 ** 2) * (m2 ** 2), (m1 ** 3) * (
                m2 ** 2), (m1 ** 2) * (m2 ** 3)
    l_3 = (l_1 * (l ** 4) + l_2 * m2 * (l ** 3))

    a = (l_2 * l) / mm
    b = ((l ** 3) * l_1 + l_2 * m2 * (l ** 2)) / m2m2
    c = (l_1 * (l ** 2)) / mm2
    d = l_3 / m2m3
    e = (l_1 * (l ** 3)) / mm3
    f = l_3 / m3m2
    g = (l / m2)
    p0 = (1 + a + g + b + c + d + e + f) ** (-1)
    p11 = a * p0
    p12 = g * p0
    p21 = b * p0
    p22 = c * p0
    p31 = f * p0
    p32 = (d + e) * p0

    p1 = p11 + p12
    p2 = p21 + p22
    p3 = p31 + p32

    final_probabilities = [p0, p1, p2, p3]
    return final_probabilities


def get_theoretical_queue_probability(final_probabilities):
    return final_probabilities[1] + final_probabilities[2]


def get_theoretical_reject_probability(final_probabilities):
    return final_probabilities[-1]


def get_theoretical_relative_bandwidth(reject_probability):
    return 1 - reject_probability


def get_theoretical_absolute_bandwidth(relative_bandwidth, _lambda):
    return relative_bandwidth * _lambda


def get_theoretical_average_queue_items_count(final_probabilities):
    return 1 * final_probabilities[1] + 2 * final_probabilities[2] + 3 * final_probabilities[3]


def get_theoretical_average_queue_system_items_count(final_probabilities):
    return 1 * final_probabilities[2] + 2 * final_probabilities[3]


def get_theoretical_average_queue_items_time(reject_probability, _lambda):
    return reject_probability / _lambda


def g(mu1, mu2):
    return np.random.exponential(1 / ((1 / mu1 + 1 / mu2) ** -1))


def get_theoretical_average_queue_system_items_time(average_queue_system_items_count, _lambda):
    return average_queue_system_items_count / _lambda


def get_theoretical_info(_lambda, mu1, mu2):
    final_probabilities = get_theoretical_final_probabilities(_lambda, mu1, mu2)
    queue_probability = get_theoretical_queue_probability(final_probabilities)
    reject_probability = get_theoretical_reject_probability(final_probabilities)
    relative_bandwidth = get_theoretical_relative_bandwidth(reject_probability)
    absolute_bandwidth = get_theoretical_absolute_bandwidth(relative_bandwidth, _lambda)
    average_queue_items_count = get_theoretical_average_queue_items_count(final_probabilities)
    average_queue_system_items_count = get_theoretical_average_queue_system_items_count(final_probabilities)
    average_queue_items_time = get_theoretical_average_queue_items_time(average_queue_items_count, _lambda)
    average_queue_system_items_time = get_theoretical_average_queue_system_items_time(average_queue_system_items_count,
                                                                                      _lambda)
    return final_probabilities, \
           queue_probability, \
           reject_probability, \
           relative_bandwidth, \
           absolute_bandwidth, \
           average_queue_system_items_count, \
           average_queue_items_count, \
           average_queue_system_items_time, \
           average_queue_items_time,


class QueueSystem:
    def __init__(self, env, n, m, l, mu1, mu2):
        self.n = n
        self.m = m
        self._lambda = l
        self.mu1 = mu1
        self.mu2 = mu2

        self.counts = []
        self.times = []
        self.queue_counts = []
        self.queue_times = []

        self.serve_items = []
        self.reject_items = []

        self.env = env
        self.resources = sp.Resource(env, n)

    def erlang(x, k, X2):
        return (((X2 ** k) * (x ** (k - 1))) / (np.math.factorial(k - 1))) * np.exp(-X2 * x)

    def serve(self):
        value = generate_erlang_time(self._lambda, self.mu1, self.mu2)
        yield self.env.timeout(value)

    def wait(self):
        yield self.env.timeout(100000)

    def get_workload(self):
        return self.resources.count

    def get_queue_len(self):
        return len(self.resources.queue)

    def start(self, action):
        while True:
            yield self.env.timeout(np.random.exponential(1 / self._lambda))
            self.env.process(action(self))


def get_experimental_final_probabilities(queue_system: QueueSystem):
    items = np.array(queue_system.reject_items + queue_system.serve_items)
    return [(len(items[items == i]) / len(items)) for i in range(1, queue_system.n + queue_system.m + 2)]


def get_experimental_queue_probability(queue_system: QueueSystem):
    items = np.array(queue_system.reject_items + queue_system.serve_items)
    return np.sum([(len(items[items == i]) / len(items)) for i in range(1, queue_system.n + queue_system.m + 2) if
                   i > queue_system.n and i < queue_system.n + queue_system.m + 1])


def get_experimental_reject_probability(queue_system: QueueSystem):
    items = np.array(queue_system.reject_items + queue_system.serve_items)
    return (len(items[items == queue_system.n + queue_system.m + 1]) / len(items))


def get_experimental_relative_bandwidth(queue_system: QueueSystem):
    return 1 - get_experimental_reject_probability(queue_system)


def get_experimental_absolute_bandwidth(queue_system: QueueSystem):
    return get_experimental_relative_bandwidth(queue_system) * queue_system._lambda


def get_experimental_average_queue_items_count(queue_system: QueueSystem):
    return st.mean(queue_system.queue_counts)


def get_experimental_average_queue_system_items_count(queue_system: QueueSystem):
    return st.mean(queue_system.counts)


def get_experimental_average_queue_items_time(queue_system: QueueSystem):
    return st.mean(queue_system.queue_times)


def get_experimental_average_queue_system_items_time(queue_system: QueueSystem):
    return st.mean(queue_system.times)


def get_experimental_info(queue_system: QueueSystem):
    final_probabilities = get_experimental_final_probabilities(queue_system)
    queue_probability = get_experimental_queue_probability(queue_system)
    reject_probability = get_experimental_reject_probability(queue_system)
    relative_bandwidth = get_experimental_relative_bandwidth(queue_system)
    absolute_bandwidth = get_experimental_absolute_bandwidth(queue_system)
    average_queue_items_count = get_experimental_average_queue_items_count(queue_system)
    average_queue_system_items_count = get_experimental_average_queue_system_items_count(queue_system)
    average_queue_items_time = get_experimental_average_queue_items_time(queue_system)
    average_queue_system_items_time = get_experimental_average_queue_system_items_time(queue_system)
    return final_probabilities, \
           queue_probability, \
           reject_probability, \
           relative_bandwidth, \
           absolute_bandwidth, \
           average_queue_items_count, \
           average_queue_system_items_count, \
           average_queue_items_time, \
           average_queue_system_items_time


def serve(queue_system: QueueSystem):
    queue_len = queue_system.get_queue_len()
    qn_count = queue_system.get_workload()
    with queue_system.resources.request():
        queue_current_len = queue_system.get_queue_len()
        qn_current_count = queue_system.get_workload()
        queue_system.queue_counts.append(queue_len)
        queue_system.counts.append(queue_len + qn_count)
        if queue_current_len <= queue_system.m:
            start = queue_system.env.now
            queue_system.queue_times.append(queue_system.env.now - start)
            yield queue_system.env.process(queue_system.serve())
            queue_system.serve_items.append(queue_current_len + qn_current_count)
            queue_system.times.append(queue_system.env.now - start)
        else:
            queue_system.reject_items.append(queue_system.n + queue_system.m + 1)
            queue_system.times.append(0)
            queue_system.queue_times.append(0)


def generate_erlang_time(l, k, X2):
    return g(k, X2)



def test(_lambda, mu1, mu2, time):
    env = sp.Environment()
    queue_system = QueueSystem(env, 1, 2, _lambda, mu1, mu2)
    env.process(queue_system.start(serve))
    env.run(until=time)

    theoretical_info = (get_theoretical_info(_lambda, mu1, mu2))
    experimental_info = (get_experimental_info(queue_system))
    return theoretical_info, experimental_info


def run_tests(test_to_run):
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    field_names = [
        "Вероятность образования очереди",
        "Вероятность отказа",
        "Относитальная пропускная способность",
        "Абсолютная пропускная способность",
        "Среднее число элементов в очереди",
        "Среднее число элементов в СМО",
        "Среднее время пребывания элемента в очереди",
        "Среднее время пребывания элемента в СМО"]


    plt.style.use('default')
    completed_test = test(test_to_run[0], test_to_run[1], test_to_run[2], test_to_run[3])

    theoretical_probabilities, experimental_probabilities = np.array(completed_test[0][0]), np.array(
        completed_test[1][0])
    nums = np.arange(len(theoretical_probabilities))
    width = 0.3
    rects1 = ax[0].bar(nums - width / 2, theoretical_probabilities, width, label="Теоретические")
    rects2 = ax[0].bar(nums + width / 2, experimental_probabilities, width, label="Эмпирические")
    ax[0].legend()
    ax[0].bar_label(rects1, padding=3, rotation='vertical')
    ax[0].bar_label(rects2, padding=3, rotation='vertical')
    ax[0].set_title(f'Система работала на протяжении {test_to_run[-1]} у.е.')

    fig.tight_layout()

    theoretical_info, experimental_info = completed_test[0][1:], completed_test[1][1:]
    for j, f in enumerate(field_names):
        if (type(experimental_info[j]) != list):
            print(field_names[j])
            print(
                f"теор  - {np.round(theoretical_info[j], 5)} - {np.round(experimental_info[j], 5)} - эмпир, разница - {np.round(abs(experimental_info[j] - theoretical_info[j]), 5)}");
    nums = np.arange(len(theoretical_info))
    rects1 = ax[1].bar(nums - width / 2, theoretical_info, width, label="Теоретические")
    rects2 = ax[1].bar(nums + width / 2, experimental_info, width, label="Эмпирические")
    ax[1].legend()
    ax[1].bar_label(rects1, padding=3, rotation='vertical')
    ax[1].bar_label(rects2, padding=3, rotation='vertical')
    ax[1].set_title(f'Сравнение характеристик системы')
    plt.show()


tests = [2, 6, 12, 55]

# sheetX = pd.ExcelFile(r"input.xlsx").parse(0)
#tests = sheetX.loc[:].values
run_tests(tests)

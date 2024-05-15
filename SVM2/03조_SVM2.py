# 03조 (임주형,이세비,최하은)

"""
SVM2 - seed가 같을 때 같은 난수를 발생시키는지 확인하기 위한 실험
=> 하단의 출력문을 통해 결과 확인
"""

import numpy as np

# 같은 시드를 사용하여 난수를 생성
seed = 1234
random_state1 = np.random.RandomState(seed)
random_numbers1 = random_state1.rand(5)

random_state2 = np.random.RandomState(seed)
random_numbers2 = random_state2.rand(5)

print("같은 시드를 사용한 첫 번째 난수 배열:", random_numbers1)
print("같은 시드를 사용한 두 번째 난수 배열:", random_numbers2)

# 결과 비교
same_seed_same_output = np.array_equal(random_numbers1, random_numbers2)
print("같은 시드를 사용했을 때 결과가 동일한가요?", same_seed_same_output)

# 다른 시드를 사용하여 난수를 생성
random_state3 = np.random.RandomState(5678)
random_numbers3 = random_state3.rand(5)

random_state4 = np.random.RandomState(8765)
random_numbers4 = random_state4.rand(5)

print("다른 시드를 사용한 첫 번째 난수 배열:", random_numbers3)
print("다른 시드를 사용한 두 번째 난수 배열:", random_numbers4)

# 결과 비교
different_seed_different_output = np.array_equal(random_numbers3, random_numbers4)
print("다른 시드를 사용했을 때 결과가 동일한가요?", different_seed_different_output)

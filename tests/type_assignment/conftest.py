import pytest

@pytest.fixture
def tree_fixture():
    return {
        'hierarchy': ['class', 'subclass', 'cluster'],
        'class': {
            'A': set(['bb', 'cc']),
            'B': set(['aa', 'dd', 'ee']),
            'C': set(['ff'])
        },
        'subclass': {
            'aa': set(['1', '3', '4']),
            'bb': set(['2',]),
            'cc': set(['0', '5', '6']),
            'dd': set(['8']),
            'ee': set(['7', '9']),
            'ff': set(['10', '11', '12'])
        },
        'cluster': {
            '0': [0, 1, 2],
            '1': [3, 5],
            '2': [4, 6, 7],
            '3': [8, 11],
            '4': [9, 12],
            '5': [10, 13],
            '6': [14,],
            '7': [15, 16, 18],
            '8': [17, 20],
            '9': [19, 21, 22],
            '10': [23, 24],
            '11': [25,],
            '12': [26, 27]
        }
    }

from typing import Any, List


def hello(name: str = None) -> str:
    if name == '' or name == None:
        return 'Hello!'

    return f'Hello, {name}!'


def int_to_roman(num: int) -> str:
    roman_numerals = {
        'M': 1000, 'CM': 900, 'D': 500, 'CD': 400,
        'C': 100, 'XC': 90, 'L': 50, 'XL': 40,
        'X': 10, 'IX': 9, 'V': 5, 'IV': 4, 'I': 1
    }

    if num == 0:
        return ''

    for numeral, value in zip(roman_numerals.keys(), roman_numerals.values()):
        if num >= value:
            return numeral + int_to_roman(num - value)


def longest_common_prefix(strs_input: List[str]) -> str:
    prefix = ''
    for prev, cur in zip(strs_input, strs_input[1:]):
        prev = prev.lstrip(' \t\n')
        cur = cur.lstrip(' \t\n')

        for i in range(0, min(len(prev), len(cur)) + 1):
            if not cur.startswith(prev[:i + 1]):
                if i == 0:
                    return ''
                if prefix == '' or len(prefix) > i - 1:
                    prefix = prev[:i]

        if prefix == '' or len(prefix) > i - 1:
            prefix = prev[:i]

    return prefix

def is_prime(n):
    if n <= 1:
        return False

    for i in range(2, int(n**(1/2)) + 1):
        if n % i == 0:
            return False

    return True


def primes() -> int:
    n = 0
    while True:
        if is_prime(n):
            yield n
        n += 1


class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = None):
        self.total_sum = total_sum

        if balance_limit == None:
            self.balance_limit = None
        self.balance_limit = balance_limit
    
    def __call__(self, sum_spent: int):
        if sum_spent <= self.total_sum:
            self.total_sum -= sum_spent
            print(f'You spent {sum_spent} dollars.')
        else:
            raise ValueError(f'Not enough money to spend {sum_spent} dollars.')
    
    def __str__(self):
        return 'To learn the balance call balance.'

    @property
    def balance(self):
        if self.balance_limit == None:
            return self.total_sum

        if self.balance_limit > 0:
            self.balance_limit -= 1
            return self.total_sum

        raise ValueError('Balance check limits exceeded.')

    
    def put(self, sum_put):
        self.total_sum += sum_put
        print(f'You put {sum_put} dollars.')
    
    def __add__(self, other):
        return BankCard(self.total_sum + other.total_sum, max(self.balance_limit, other.balance_limit))


codegen_solution = """
```python
def can_transform_to_abc(s):
    target = "abc"
    
    # If already in correct order, no swap needed
    if s == target:
        return "YES"
    
    # Check if exactly one swap can convert s to "abc"
    differences = [(s[i], target[i]) for i in range(3) if s[i] != target[i]]
    
    # A single swap means exactly two characters must be out of place
    if len(differences) == 2 and differences[0] == differences[1][::-1]:
        return "YES"
    
    return "NO"

def process_test_cases():
    t = int(input().strip())
    results = []
    for _ in range(t):
        s = input().strip()
        results.append(can_transform_to_abc(s))
    print("\\n".join(results))

# Run the function to process input
process_test_cases()
```
"""

codegen_solution2 = """

def solve_case():
    n = int(input())
    digits = list(map(int, input().split()))
    
    max_product = 0
    
    # Try adding 1 to each position
    for i in range(n):
        # Calculate product with current digit increased by 1
        product = 1
        for j in range(n):
            if j == i:
                product *= (digits[j] + 1)
            else:
                product *= digits[j]
        max_product = max(max_product, product)
    
    return max_product

# Read number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    print(solve_case())

"""


codegen_solution3 = """
The following are the solutions for the problem:

```python
def solve_case():
    n, k = map(int, input().split())
    s = input()
    
    # If no black cells, return 0
    if 'B' not in s:
        return 0
    
    operations = 0
    i = 0
    
    while i < n:
        # Find next black cell
        if s[i] == 'B':
            # Cover k cells starting from this black cell
            operations += 1
            i += k
        else:
            i += 1
            
    return operations

# Read number of test cases
t = int(input())

# Process each test cases
for _ in range(t):
    print(solve_case())
```
"""

test_output_prediction_solution1 = """
To determine the output of the test case, we need to analyze the provided strings in the `details` list and count how many passengers are strictly more than 60 years old.

The age is represented by the 12th and 13th characters of each string. We will extract these characters, convert them to an integer, and check if they are greater than 60.

Let's break down the provided input:

1. **"7868190130M7522"**: 
   - Age: "75" (which is greater than 60)
   
2. **"5303914400F9211"**: 
   - Age: "92" (which is greater than 60)
   
3. **"9273338290F4010"**: 
   - Age: "40" (which is not greater than 60)

Now, we count the passengers who are older than 60:
- The first string represents a passenger who is 75 years old.
- The second string represents a passenger who is 92 years old.
- The third string represents a passenger who is 40 years old.

Thus, there are 2 passengers who are strictly more than 60 years old.

The assertion statement will be:

```python
assert countSeniors(["7868190130M7522", "5303914400F9211", "9273338290F4010"]) == 2
```
"""

test_output_prediction_solution2 = """
To determine the output for the given test case, we need to analyze the provided strings in the `details` list.

1. The first string is `"1313579440F2036"`:
   - The age is represented by the 12th and 13th characters, which are `"20"`. This indicates that the passenger is 20 years old.

2. The second string is `"2921522980M5644"`:
   - The age is represented by the 12th and 13th characters, which are `"56"`. This indicates that the passenger is 56 years old.

Now, we need to count how many passengers are strictly more than 60 years old. In this case, both passengers are under 60 years old.

Thus, the expected output for the assertion should be `0`.

Here is the complete assertion statement:

```python
assert countSeniors(["1313579440F2036", "2921522980M5644"]) == 0
```
"""

test_output_prediction_solution3 = """
To calculate the output for the given input `[[7, 2, 1], [6, 4, 2], [6, 5, 3], [3, 2, 1]]`, we will follow the steps outlined in the problem statement.

1. **Initial Matrix**: 
   ```
   [7, 2, 1]
   [6, 4, 2]
   [6, 5, 3]
   [3, 2, 1]
   ```

2. **Step 1**: From each row, select the largest number and remove it:
   - From the first row `[7, 2, 1]`, the largest number is `7`.
   - From the second row `[6, 4, 2]`, the largest number is `6`.
   - From the third row `[6, 5, 3]`, the largest number is `6`.
   - From the fourth row `[3, 2, 1]`, the largest number is `3`.

   The removed numbers are `[7, 6, 6, 3]`.

3. **Step 2**: Identify the highest number among the removed numbers:
   - The highest number is `7`.

4. **Step 3**: Add that number to the score:
   - The score is now `7`.

5. **Repeat**: The matrix is now:
   ```
   [2, 1]
   [4, 2]
   [5, 3]
   [2, 1]
   ```

   - From the first row `[2, 1]`, the largest number is `2`.
   - From the second row `[4, 2]`, the largest number is `4`.
   - From the third row `[5, 3]`, the largest number is `5`.
   - From the fourth row `[2, 1]`, the largest number is `2`.

   The removed numbers are `[2, 4, 5, 2]`.

   - The highest number is `5`, so we add `5` to the score: `7 + 5 = 12`.

6. **Repeat**: The matrix is now:
   ```
   [1]
   [2]
   [3]
   [1]
   ```

   - From the first row `[1]`, the largest number is `1`.
   - From the second row `[2]`, the largest number is `2`.
   - From the third row `[3]`, the largest number is `3`.
   - From the fourth row `[1]`, the largest number is `1`.

   The removed numbers are `[1, 2, 3, 1]`.

   - The highest number is `3`, so we add `3` to the score: `12 + 3 = 15`.

7. **Repeat**: The matrix is now empty, and we stop here.

The final score is `15`.

Now, we can write the assertion statement:

```python
assert matrixSum([[7, 2, 1], [6, 4, 2], [6, 5, 3], [3, 2, 1]]) == 15
```
"""

code_execution_solution1 = """
[ANSWER]
assert distinctDifferenceArray(a = [1, 2, 3, 4, 5]) == [-3, -1, 1, 3, 5]
[/ANSWER]
"""

code_execution_solution2 = """
[ANSWER]
assert distinctDifferenceArray(nums = [1, 2, 3, 4, 5]) == [-3, -1, 1, 3, 5]
[/ANSWER]
"""

code_execution_solution3 = """
[ANSWER]
assert distinctDifferenceArray(nums = [1, 2, 3, 4, 5]) == [-3, -1, 1, 3, 5]
[/ANSWER]
"""
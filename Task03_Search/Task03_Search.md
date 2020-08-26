# Task03 查找

Datawhale2群 102-StarLEE

## 一. 查找表

### 查找有无--set

集合`set`是一个无序的不重复元素序列可以使用大括号`{ }`或`set()`函数创建集合（注意空集合必须用`set()`创建），set只储存key，不储存对应的value，且set中的key不允许重复，所以很适用于查找特定元素是否在特定容器中。

### 查找对应关系--dict

字典`dict`是一种可变容器模型，可储存任意类型对象。字典中键必须是唯一的，但值则不必。值可以取任何数据类型，但键必须是不可变的，如字符串，数字。字典适用于对特定元素计数等任务

### 改变映射关系--map

`map()`函数使用方法为`map(function, iterable)`，函数的主要作用是将一个迭代对象中的每个值逐个放入用户指定的函数中进行处理，如果传入多个迭代对象，那么指定的函数也必须接受对等数量的参数，最终返回的结果数量，以多个迭代对象中数量最少的那个为准。

### LeetCode例题

**No.349 Intersection Of Two Arrays 1**

题目：给定两个数组，求交集

思路：交集中的元素是唯一的，故可以遍历数组，用set来储存intersection

算法实现：

```python
class Solution:
    def intersection(self, nums1, nums2):
        if not nums1 or not nums2:
            return []
        ans = set()
        for i in nums1:
            if i in nums2:
                ans.add(i)
        return ans
```

**No.350 Intersection Of Two Arrays 2**

题目：给定两个数组，求交集(元素计次数)

思路：一个collections.Counter是一个dict的子类，用于计数可哈希对象。它是一个集合，元素像键一样存储，计数存为值。该方法对于统计数组中不同数字个数。

算法实现：

```python
class Solution:
    def intersect(self, nums1, nums2):
        if not nums1 or not nums2:
            return []
        
        ans = []
        n2 = collections.Counter(nums2)
        
        for i in nums1:
            if i in n2 and n2[i]:
                ans.append(i)
                n2[i] -= 1
        return ans
```

**No.242 Valid Anagram**

题目：给定两个字符串s和t，判断t是否为s的字母异位词

思路：

s和t具有相同数量的字母，则两个互为字母异位词；collections.Counter也可以对字符串中不同字符的计数。

算法实现：

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        return collections.Counter(s) == collections.Counter(t)
```

**No.202 Happy Number**

题目：给定一个数字，判断是否为happy number

思路：

循环有两个结果：最终等于1和陷入死循环，判断死循环的依据就是记录每次结果

算法实现：

```python
class Solution:
    def sumOfDigits(self, n):
        res = 0
        while n >= 10:
            res += (n%10)**2
            n = n // 10
        res += (n%10)**2
        return res
    
    def isHappy(self, n: int) -> bool:
        record = {n}
        while True:
            n = self.sumOfDigits(n)
            # print(n)
            if n == 1:
                return True
            if n in record:
                return False
            record.add(n)
```

**No.290 Word Pattern**

题目：给定一个数字，判断是否为happy number

思路：

循环有两个结果：最终等于1和陷入死循环，判断死循环的依据就是记录每次结果

算法实现：

```python
class Solution:
    def sumOfDigits(self, n):
        res = 0
        while n >= 10:
            res += (n%10)**2
            n = n // 10
        res += (n%10)**2
        return res
    
    def isHappy(self, n: int) -> bool:
        record = {n}
        while True:
            n = self.sumOfDigits(n)
            # print(n)
            if n == 1:
                return True
            if n in record:
                return False
            record.add(n)
```

**No.205 Isomorphic Strings**

题目：判断两字符串是否同构

思路：

作为上题的补充循环有两个结果：最终等于1和陷入死循环，判断死循环的依据就是记录每次结果

算法实现：











****

## 动态规划编程

适用于动态规划的问题需要具有两个关键属性：***最优子结构***和***重叠子问题***。如果可以通过将最优解决方案与非重叠子问题组合来解决问题，则该策略成为“分治”。这也是为什么并归排序和快速排序未归类为动态规划的原因

*最优子结构*是指可以通过组合子问题的最优解来获得给定优化问题的解。通常借助递归来描述最优子结构。例如给定一个图G = (V, E)，从点u到点v的最短路径p表现出最优子结构：在这个最短路径p上取任意中间点w。如果p确实为最短路径，则可以将其分为u到w的子路径p1和w到v的子路径p2，这些子路径实际上就是对应顶点之间的最短路径。因此可以指定一种以递归方式寻找最短路径的解决方案，这就是Bellman-Ford算法。

*重叠子问题*意味着子问题的空间很小，任何解决该问题的递归算法都应该一遍又一遍地解决相同的子问题，而不是生成新的子问题。例如生成斐波那契数列的递归公式：F_i = F_i-1 + F_i-2，基本情况为F_1 = F_2 = 1。即使子问题总数很小，但如果这样递归地解决方案，我们最终还是会一遍又一遍解决相同问题。动态规划考虑到这一事实，并且只解决了每个子问题一次。

可以通过两种方式来实现：

**自上而下：**这是任何问题递归的直接结果。如果可以使用子问题的解来递归地提出任何问题的解，并且子问题重叠，则可以将子问题的解存储在表中。每当尝试解决新的子问题时，都可首先查看表以判断是否问题已被解决过。如果已记录过解，则直接使用，否则解决问题并添加到表中。

**自下而上：**按照子问题递归地提出问题的解后，可以通过自下而上的方式重新构建问题：尝试先解决子问题，然后使用其解来构建并提出更大的子问题的解。通常也以表的形式通过使用较小子问题的解迭代生成越来越大的子问题的解来完成操作。例如，若我们知到了F_i-1和F_i-2的值，则可以直接得到F_i的值。

****

## 算法实现

动态规划问题的一般形式就是求最值。动态规划是运筹学的一种最优化方法，不过在计算机问题上应用也很多

求最值的核心问题是**穷举**。因为要求最值，肯定要把所有可行的答案穷举出来，然后在其中找最值。但是动态规划的穷举有点特别，因为这类问题存在重叠子问题，如果暴力穷举的话效率会极其低下，所以需要「备忘录」或者「DP table」来优化穷举过程，避免不必要的计算。而且，动态规划问题一定会具备最优子结构，才能通过子问题的最值得到原问题的最值。

另外，虽然动态规划的核心思想就是穷举求最值，但是问题可以千变万化，穷举所有可行解并不是一件容易的事，只有列出正确的状态转移方程才能正确地穷举，在实际的算法问题中，写出状态转移方程是较为困难的。

可以遵循一个流程：明确 base case → 明确状态 → 明确选择 → 定义 dp 数组/函数的含义。按上面的流程走，可以套下方框架：

```python
# 初始化 base case
dp[0][0]...[0] = base
# 状态转移
for 状态1 in 状态1所有取值:
    ...
    	for 状态n in 状态n所有取值:
            dp[状态1][状态2]...[状态n] = 最值(选择1, 选择2,...)
```

****

## LeetCode例题

**No.674 最长连续递增序列**

题目：给定未排序整数数组，找到最长且连续的递增序列

思路：

+ 动态规划状态：dp[i]为以nums[i]结尾的最长递增子序列长度
+ 状态转移方程：nums[i]前后不连续递增，长度为1；nums[i]前后连续递增，`dp[i] = dp[i-1] + 1`
+ 边界条件：初始化元素均为1的数组
+ 输出状态：返回`max(dp)`

算法实现：

```python
class Solution:
    def findLengthOfLCIS(self, nums: List[int]) -> int:
        if not nums:
            return 0
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            if nums[i] > nums[i-1]:
                dp[i] = dp[i-1] + 1
            else:
                 dp[i] = 1
        return max(dp)
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task02_Dynamic%20programming/pics/1_674.png?raw=true" style="zoom:80%;" />

**No.5 最长回文子串**

题目：给定字符串s，找到s中最长的回文子串

思路：

+ 动态规划状态：`dp[i][j]`为`s[i:j]`是否为回文串
+ 状态转移方程：首尾字符不等，为False；首尾字符相等，看子串

`dp[i][j] =  dp[i+1][j-1] and s[i] == s[j]`

+ 边界条件：元素为False的list of list

+ 输出状态：返回回文串中最长的

算法实现：

```python
def longestPalindrome(self, s: str) -> str:
    n = len(s)
    dp = [[False] * n for _ in range(n)]
    ans = ""
    # 枚举子串的长度1 ~ n
    for l in range(n):
	  	# 枚举子串的起始位置i，通过j=i+l得到结束位置
        for i in range(n):
            # 得到结束位置
            j = i + l
            # 结束位置超范围，跳出
            if j >= len(s):
                break
            # 长度为1子串为回文串
            if l == 0:
                dp[i][j] = True
            # 长度为2子串看两字符是否相等
            elif l == 1:
                dp[i][j] = (s[i] == s[j])
            # 状态转移方程
            else:
                dp[i][j] = (dp[i+1][j-1] and s[i] == s[j])
            # 存在更长的回文串则更新ans
            if dp[i][j] and l + 1 > len(ans):
                ans = s[i:j+1]
    return ans
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task02_Dynamic%20programming/pics/2_5.png?raw=true" style="zoom:80%;" />

**No.516 最长回文子序列**

题目：给定字符串s，找到s中最长的回文子序列

思路：

+ 动态规划状态：`dp[i][j]`为`s[i:j]`中最长回文序列长度

+ 状态转移方程：首尾字符不等

  ​							`dp[i][j] =  max(dp[i+1][j], dp[i][j-1])`

  ​							首尾字符相等

  ​							`dp[i][j] =  dp[i+1][j-1] + 2`

+ 边界条件：`dp[i][i] = 1`

+ 输出状态：返回`dp[0][n-1]`

算法实现：

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        for i in range(n):
            dp[i][i] = 1
            for j in range(i - 1, -1, -1):
                if s[j] == s[i]:
                    dp[j][i] = 2 + dp[j + 1][i - 1]
                else:
                    dp[j][i] = max(dp[j + 1][i], dp[j][i - 1])
        return dp[0][n - 1]
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task02_Dynamic%20programming/pics/3_516.png?raw=true" style="zoom:80%;" />

**No.72 编辑距离**

题目：给定两个单词word1，word2，计算word1转换为word2所使用的最少操作数

思路：

+ 动态规划状态：`dp[i][j]`为A前i个字母与B前j个字母的编辑距离
+ 状态转移方程：

AB最后字符相同：

```python
dp[i][j] = min(dp[i][j-1]+1, dp[i-1][j]+1, dp[i-1][j-1])
		 = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1]-1)
```

字符不同：

```python
dp[i][j] = 1 + min(dp[i][j-1], dp[i-1][j], dp[i-1][j-1])
```

+ 边界条件：`dp[i][0] = i`和`dp[0][j] = j`
+ 输出状态：返回`dp[n][m]`

算法实现：

```python
class Solution:
    def minDistance(self, word1, word2):
        n = len(word1)
        m = len(word2)
        
        # 有一个字符串为空串
        if n * m == 0:
            return n + m
        
        # dp数组
        dp = [ [0] * (m + 1) for _ in range(n + 1)]
        
        # 边界状态初始化
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # 计算dp值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = dp[i - 1][j] + 1
                down = dp[i][j - 1] + 1
                left_down = dp[i - 1][j - 1] 
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                dp[i][j] = min(left, down, left_down)
        
        return dp[n][m]
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task02_Dynamic%20programming/pics/4_72.png?raw=true" style="zoom:80%;" />

**No.198 打家劫舍**

题目：每屋有现金，最高打劫金额（不能抢相邻屋子）

思路：

+ 动态规划状态：`dp[i]`为前i个屋子抢劫最高金额
+ 状态转移方程：

```python
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

+ 边界条件：`dp[0] = nums[0]`和`dp[1] = max(nums[0], nums[1])`
+ 输出状态：返回`dp[n-1]`

算法实现：

```python
class Solution:
    def minDistance(self, word1, word2):
        n = len(word1)
        m = len(word2)
        
        # 有一个字符串为空串
        if n * m == 0:
            return n + m
        
        # dp数组
        dp = [ [0] * (m + 1) for _ in range(n + 1)]
        
        # 边界状态初始化
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # 计算dp值
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                left = dp[i - 1][j] + 1
                down = dp[i][j - 1] + 1
                left_down = dp[i - 1][j - 1] 
                if word1[i - 1] != word2[j - 1]:
                    left_down += 1
                dp[i][j] = min(left, down, left_down)
        
        return dp[n][m]
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task02_Dynamic%20programming/pics/5_198.png?raw=true" style="zoom:80%;" />

**No.213 打家劫舍II**

题目：每屋有现金，屋子围成圈，最高打劫金额（不能抢相邻屋子）

思路：

+ 动态规划状态：`dp[i]`为前i个屋子抢劫最高金额
+ 状态转移方程：

```python
dp[i] = max(dp[i-1], dp[i-2] + nums[i])
```

+ 边界条件：`dp[0] = nums[0]`和`dp[1] = max(nums[0], nums[1])`
+ 输出状态：返回`dp[n-1]`

算法实现：

```python
class Solution:
    def rob(self, nums: List[int]) -> int:
        def main(nums):
            n=len(nums)
            if n==0:return 0
            if n==1:return nums[0]
            pre=nums[0]
            cur=max(nums[0],nums[1])
            for i in range(2,n):
                temp=cur
                cur=max(pre+nums[i],cur)
                pre=temp
            return cur
        n=len(nums)
        if n==0:return 0
        if n==1:return nums[0]
        if n==2:return max(nums[0],nums[1])
        return max(main(nums[:n-1]),main(nums[1:n]))
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task02_Dynamic%20programming/pics/6_213.png?raw=true" style="zoom:80%;" />
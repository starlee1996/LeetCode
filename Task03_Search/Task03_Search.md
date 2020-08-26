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










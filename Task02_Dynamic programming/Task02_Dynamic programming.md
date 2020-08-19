# Task02 动态规划

Datawhale2群 102-StarLEE

## 动态规划介绍

动态规划（dynamic programming）







是一种基于多分支递归的算法。分治算法就是把一个复杂的问题分成两个或更多的相同或相似的子问题，直到最后子问题可以简单的直接求解，原问题的解即子问题的解的合并。

分治是解决很多高效算法的基础，如排序算法（快速排序，并归排序），快速乘法算法（Karatsuba算法），最近点对问题，语法分析（top-down parsers）以及傅里叶变换（FFT）。

理解并设计分治算法是一项较为复杂的事情，需对要解决的基本问题有很好了解。有如通过归纳证明定理，为了使得递归进行，通常需要使用一个较为概括或复杂的问题替换原始问题。并且分治往往没有系统性的方法来合适地概括问题。

分治法通常用数学归纳法来验证，且他的复杂度多以解递归关系式来得到。

![](https://github.com/starlee1996/LeetCode/blob/master/Task01_Divide-and-conquer/pics/1_%E5%88%86%E6%B2%BB%E7%A4%BA%E6%84%8F%E5%9B%BE.png?raw=true)

****

## 分治优点

**解决复杂问题：**分治法是解决困难问题的有力工具，它所需要的只是将问题分解为子问题，解决较为容易的子问题后，将其合并产生原问题解。

**算法效率：**分治法的范式通常有助于发现高效算法。分治算法是Karatsuba快速乘法算法，快速排序，并归排序，用于矩阵乘法的Strassen算法以及快速傅里叶变换算法等的关键思想。

**同步性：**分治算法适用于具有多处理器的系统。尤其是共享内存系统，因为可以在不同处理器上解决不同子问题，它无需预先计划处理器之间的数据通信。

**内存存取：**分治算法可有效利用内存缓存。一旦子问题足够小，原则上就可以在缓存中解决所有的子问题。

****

## 算法实现

**递归：**分治法是作为递归过程实现的。当前正在解决的子问题会自动存储在过程调用的栈中。

**显式栈：**分治法也可以通过将部分子问题存储在某些显式数据结构（栈，队列）中的非递归程序来实现。这种方法在选择下一个子问题时提供了更大的自由度，这一功能在某些应用程序中很重要。

**栈的容量：**在分治法的递归实现中，必须确保为递归栈分配了足够的内存，否则执行可能会由于栈的溢出而失败。高效的分治法通常将具有相对较小的递归深度，例如对*n*个元素进行快速排序，则最多只需要log2(n)次调用递归过程。可以通过最小化递归过程的参数和内部变量，或使用显示栈结构来减少栈溢出的风险。

**子问题选择：**选择子问题具有很大的自由度，当子问题小到可以直接解决时便终止递归

在每一层递归上有三个步骤：

1. 分：将问题分解为若干规模较小，相对独立，与原问题形式相同的子问题
2. 治：若子问题规模小且易解决，则直接解。否则，递归地解决各子问题
3. 合：将各子问题的解合并为原问题的解

伪代码：

```python
def divideConquer(prob, params):
    # 终止条件
    if prob is None:
        return result
    # 准备数据
    data = prepareData(prob)
    # 分
    subprobs = splitProblem(prob, data)
    # 治
    subres[1] = divideConquer(subprobs[0],ps)
    subres[2] = divideConquer(subprobs[1],ps)
    subres[3] = divideConquer(subprobs[2],ps)
    # 对子结果进行合并 得到最终结果
    result = process_result(subres)
```

****

## LeetCode例题

**No.169 多数元素**

题目：给定大小为*n*数组，找到其中众数

思路：

+ 分：把数组不断细分为若干不相交的子数组
+ 治：得到每个子数组的众数（显然长度为1的数组众数就是那唯一元素）
+ 合：子数组合并得到大数组，其众数为两子数组众数中更多的那个元素

算法实现：

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # 特殊情况
        if not nums:
            return None
        
        # 确定迭代终止条件
        if len(nums) == 1:
            return nums[0]
        
        # 分治：将数组左右平分并开始迭代
        l = self.majorityElement(nums[:len(nums)//2])
        r = self.majorityElement(nums[len(nums)//2:])
        
        # 合
        return l if nums.count(l) >= nums.count(r) else r
```

![](https://github.com/starlee1996/LeetCode/blob/master/Task01_Divide-and-conquer/pics/2_169%E7%BB%93%E6%9E%9C.png?raw=true)

**No.53 最大子序和**

题目：给定整数数组`nums`，返回具有最大和连续子数组（最少一个元素）的和

思路：

+ 分：把整数数组不断细分为若干不相交的子数组
+ 治：得到每个子数组的和（显然长度为1的数组和就是那唯一元素）
+ 合：子数组合并得到大数组，其最大和为两子数组和中最大数或加和

算法实现：

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        # 特殊情况
        if not nums:
            return None
        
        # 终止条件
        if len(nums) == 1:
            return nums[0]
        
        # 分治
        l = self.maxSubArray(nums[:len(nums)//2])
        r = self.maxSubArray(nums[len(nums)//2:])
        
        
        # 动态规划dp
        # 从右至左计算左子数组最大和
        max_l = nums[len(nums)//2 - 1]
        tmp = 0
        
        for i in range(len(nums)//2 - 1, -1, -1):
            tmp += nums[i]
            max_l = max(tmp, max_l)
        
        # 从左至右计算右子数组最大和
        max_r = nums[len(nums)//2]
        tmp = 0
        
        for i in range(len(nums)//2, len(nums)):
            tmp += nums[i]
            max_r = max(tmp, max_r)
        
        return max(l, r, max_l + max_r)
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task01_Divide-and-conquer/pics/3_53%E7%BB%93%E6%9E%9C.png?raw=true" alt="53" style="zoom:80%;" />

**No.50 Pow(x,n)**

题目：实现`pow(x, n)`，即x的n次幂

思路：

+ 分：将n不断除以2
+ 治：对于每一对x，计算`x * x`来实现`pow(x, 2)`
+ 合：若n为偶数则将多个`pow(x, 2)`乘起来，若为奇数则最后多乘以一个x

算法实现：

```python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        # 特殊情况
        if n < 0:
            x = 1 / x
            n = -n
        
        # 终止条件
        if n == 0:
            return 1
        
        # 对于奇数情况
        if n % 2 == 1:
            return x * self.myPow(x, n-1)
        
        # 对于偶数情况
        return self.myPow(x*x, n/2)
```

<img src="https://github.com/starlee1996/LeetCode/blob/master/Task01_Divide-and-conquer/pics/4_50%E7%BB%93%E6%9E%9C.png?raw=true" alt="50" style="zoom:80%;" />


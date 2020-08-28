# Task03 查找2

Datawhale2群  102-StarLEE

## 二. 对撞指针（双指针）

**双指针**是指在遍历对象的过程中，不单单使用一个指针进行访问，而是使用两个相同方向（*快慢指针*）或者相反方向（*对撞指针*）的指针进行扫描，从而达到目的。换言之，双指针法充分使用了数组有序这一特征，从而在某些情况下能够简化一些运算。

### LeetCode例题

**No.1 Two Sum**

题目：给定整数数组nums，返回`nums[i]+num[j]=target`的索引值数对

暴力循环算法实现：

```python
class Solution:
    def twoSum(self, nums, target):
        if not nums:
            return []
        
        for i in range(len(nums)):
            for j in range(i+1, len(nums)):
                if nums[i]+nums[j]==target:
                    return [i,j]
```

排序+双指针算法实现：

```python
class Solution:
    def twoSum(self, nums, target):
        # 复制一份保存indexes
        ncopy = nums.copy()
        # 用于防止重复使用
        isSame = True;
        # 数组升序排列
        nums.sort()
        # 设置双指针
        l, r = 0, len(nums)-1
        # 指针开始移动
        while l < r:
            # 相等则找到答案
            if nums[l] + nums[r] == target:
                break
            # 小就左指针右移
            elif nums[l] + nums[r] < target:
                l += 1
            # 大就右指针左移
            else:
                r -= 1
        res = []
        # 遍历copy数组获取原indexes
        for i in range(len(nums)):
            # 找到指针对应数字index且未重复，计入结果
            if ncopy[i] == nums[l] and isSame:
                res.append(i)
                isSame = False
            elif ncopy[i] == nums[r]:
                res.append(i)
        return res
```

`enumerate()`函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标。

`enumerate()`算法实现：

```python
class Solution:
    def twoSum(self, nums, target):
        nums = list(enumerate(nums))
        nums.sort(key = lambda x:x[1])
        i,j = 0, len(nums)-1
        while i < j:
            if nums[i][1] + nums[j][1] > target:
                j -= 1
            elif nums[i][1] + nums[j][1] < target:
                i += 1
            else:
                if nums[j][0] < nums[i][0]:
                    nums[j],nums[i] = nums[i],nums[j]
                return nums[i][0],nums[j][0]
```

`i + j = target ~ i = target - j`，由此建立查找表

查找表算法实现：

```python
class Solution:
    def twoSum(self, nums, target):
        record = dict()
        for i in range(len(nums)):
            c = target - nums[i]
            if record.get(c) is not None:
                res = [i,record[c]]
                return res
            record[nums[i]] = i
```

**No.15 3Sum**

题目：给定数组，返回三数和等于0的下标

思路：对数组进行排序。然后遍历排序后数组：若`nums[i]>0`，后面不可能有三个数加和等于0，直接返回结果。对于重复元素：跳过，避免出现重复解。令左指针 `L=i+1`，右指针`R=n-1`，当 `L<R`时，执行循环：
当 `nums[i]+nums[L]+nums[R]==0`，执行循环，判断左界和右界是否和下一位置重复，去除重复解。并同时将L，R移到下一位置，寻找新的解。若和大于0，说明 `nums[R]`太大，R左移；若和小于0，说明`nums[L]`太小，L右移。

算法实现：

```python
class Solution:
    def threeSum(self, nums):
        n=len(nums)
        res=[]
        if(not nums or n<3):
            return []
        nums.sort()
        res=[]
        for i in range(n):
            if(nums[i]>0):
                return res
            if(i>0 and nums[i]==nums[i-1]):
                continue
            L=i+1
            R=n-1
            while(L<R):
                if(nums[i]+nums[L]+nums[R]==0):
                    res.append([nums[i],nums[L],nums[R]])
                    while(L<R and nums[L]==nums[L+1]):
                        L=L+1
                    while(L<R and nums[R]==nums[R-1]):
                        R=R-1
                    L=L+1
                    R=R-1
                elif(nums[i]+nums[L]+nums[R]>0):
                    R=R-1
                else:
                    L=L+1
        return res
```

**No.18 4Sum**

题目：给定数组，返回四数和等于的target下标

思路：四数之和与前面三数之和的思路几乎是一样的。这里其实就是在前面的基础上多添加一个指针。使用四个指针，固定最小的a和b在左边，c=b+1，d=_size-1 移动两个指针求解。保存使得`nums[a]+nums[b]+nums[c]+nums[d]==target`的解。偏大时d左移，偏小时c右移。c和d相遇时，表示以当前的a和b为最小值的解已经全部求得。`b++`进入下一轮循环b循环。当b循环结束后，`a++`进入下一轮a循环。a在最外层循环，里面嵌套b循环，再嵌套双指针c，d包夹求解。

算法实现：

```python
class Solution:
    def fourSum(self, nums, target):
        nums.sort()
        res = []
        if len(nums) < 4:
            return res
        n = len(nums)
        for a in range(n-3):
            if a>0 and nums[a]==nums[a-1]:
                continue
            for b in range(a+1, n-2):
                if b>a+1 and nums[b]==nums[b-1]:
                    continue
                c, d = b+1, n-1
                while c<d:
                    if nums[a]+nums[b]+nums[c]+nums[d]<target:
                        c += 1
                    elif nums[a]+nums[b]+nums[c]+nums[d]>target:
                        d -= 1
                    else:
                        res.append([nums[a],nums[b],nums[c],nums[d]])
                        while c<d and nums[c+1]==nums[c]:
                            c += 1
                        while c<d and nums[d-1]==nums[d]:
                            d -= 1
                        c += 1
                        d -= 1
        return res
```

**No.16 3Sum Closest**

题目：给定数组，返回三数和最接近target的和

思路：首先数组排序，进行遍历，每遍历一个值利用其下标`i`，形成一个固定值 `nums[i]`。再使用前指针指向`start=i+1` 处，后指针指向`end = len(nums)-1` 处。根据`sum = nums[i] + nums[start] + nums[end]`的结果，判断`sum`与目标 `target`的距离，如果更近则更新结果`ans`。同时判断`sum`与`target`的大小关系，因为数组有序，如果 `sum > target`则`end--`，如果`sum < target`则`start++`，如果`sum == target`则直接返回结果。

算法实现：

```python
class Solution:
    def threeSumClosest(self, nums, target):
        nums.sort()
        ans = nums[0] + nums[1] + nums[2]
        for i in range(len(nums)):
            start, end = i+1, len(nums)-1
            while start < end:
                sum_ = nums[start] + nums[end] + nums[i]
                if abs(target - sum_) < abs(target - ans):
                    ans = sum_
                if sum_ > target:
                    end -= 1
                elif sum_ < target:
                    start += 1
                else:
                    return ans
        return ans
```

**No.454 4SumII**

题目：给定数组，返回四数和等于0的组合个数

思路：初始化计数器`dic`记录数组A和B元素和，及其次数。遍历数组C和D，累加满足四数相加和为0的个数

算法实现：

```python
class Solution:
    def fourSumCount(self, A, B, C, D):
        dic = collections.Counter()
        ans = 0
        for a in A:
            for b in B:
                dic[a+b] += 1
        for c in C:
            for d in D:
                ans += dic[-c-d]
        return ans
```

**No.49 Group Anagrams**

题目：给出字符串数组，将其中可以通过颠倒字符顺序变一样的单词分组

思路：给每个字符串排序，然后计数

算法实现：

```python
class Solution:
    def groupAnagrams(self, strs):
        dic = {}
        for i in strs:
            tmp = ''.join(sorted(list(i)))
            if dic.get(tmp) is not None:
                dic[tmp].append(i)
            else:
                dic[tmp] = [i]
        return list(dic.values())
```

**No.447 Number of Boomerangs**

题目：给定平面上 n 对不同的点，找到所有元组 (i, j, k) ，其中 i 和 j 距离等于 i 和 k 距离。

思路：每次固定一个点，使用哈希表存储其他点到这个点的距离，如果存在记录次数，回旋镖的数量应为次数*（次数-1）

算法实现：

```python
class Solution:
    def numberOfBoomerangs(self, points):
        res=0
        for i in points:
            dicts={}
            for j in points:
                if i==j:
                    continue
                dicts[(i[0]-j[0])**2+(i[1]-j[1])**2]=dicts.get((i[0]-j[0])**2+(i[1]-j[1])**2,0)+1
            for i in dicts.values():
                res+=i*(i-1)
        return res
```

**No.149 Max Points on a Line**

题目：给定一个二维平面，平面上有 *n* 个点，求最多有多少个点在同一条直线上。

思路：固定一点，找其他点和这个点组成直线，统计他们的斜率

算法实现：

```python
class Solution:
    def maxPoints(self, points: List[List[int]]) -> int:
        from collections import Counter, defaultdict
        # 所有点统计
        points_dict = Counter(tuple(point) for point in points)
        # 把唯一点列举出来
        not_repeat_points = list(points_dict.keys())
        n = len(not_repeat_points)
        if n == 1: return points_dict[not_repeat_points[0]]
        res = 0
        # 求最大公约数
        def gcd(x, y):
            if y == 0:
                return x
            else:
                return gcd(y, x % y)

        for i in range(n - 1):
            x1, y1 = not_repeat_points[i][0], not_repeat_points[i][1]
            # 斜率
            slope = defaultdict(int)
            for j in range(i + 1, n):
                x2, y2 = not_repeat_points[j][0], not_repeat_points[j][1]
                dy, dx = y2 - y1, x2 - x1
                g = gcd(dy, dx)
                if g != 0:
                    dy //= g
                    dx //= g
                slope["{}/{}".format(dy, dx)] += points_dict[not_repeat_points[j]]
            res = max(res, max(slope.values()) + points_dict[not_repeat_points[i]])
        return res
```


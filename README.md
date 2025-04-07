# Cotinue Updating...

# 88. 合并两个有序数组

给你两个按 **非递减顺序** 排列的整数数组 `nums1` 和 `nums2`，另有两个整数 `m` 和 `n` ，分别表示 `nums1` 和 `nums2` 中的元素数目。

请你 **合并** `nums2` 到 `nums1` 中，使合并后的数组同样按 **非递减顺序** 排列。

注意：最终，合并后数组不应由函数返回，而是存储在数组 `nums1` 中。为了应对这种情况，`nums1` 的初始长度为 `m + n`，其中前 `m` 个元素表示应合并的元素，后 `n` 个元素为 `0` ，应忽略。`nums2` 的长度为 `n` 。

时间复杂度`O(m+n)`，空间复杂度`O(1)`。

>示例 1：
输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [**1**,**2**,2,**3**,5,6] ，其中斜体加粗标注的为 nums1 中的元素。

```python
class Solution(object):
    def merge(self, nums1, m, nums2, n):
        # 用双指针去遍历nums1和nums2，把大的放nums1的后面
        i = m - 1
        j = n - 1
        k = m + n -1

        while i >= 0 and j >= 0:
            if nums1[i] > nums2[j]:
                nums1[k] = nums1[i]
                i -= 1
            else:
                nums1[k] = nums2[j]
                j -= 1
            k -= 1
        
        # 往nums1中插，如果nums2全插完了，那么剩余的就是符合顺序的，所以只需要考虑nums2没插完的情况
        while j >= 0:
            nums1[k] = nums2[j]
            j -= 1
            k -= 1
        
        return nums1
```

---

# 27. 移除元素
给你一个数组 `nums` 和一个值 `val`，你需要 原地 移除所有数值等于 `val` 的元素。元素的顺序可能发生改变。然后返回 `nums` 中与 `val` 不同的元素的数量。

假设 `nums` 中不等于 `val` 的元素数量为 `k`，要通过此题，您需要执行以下操作：

更改 `nums` 数组，使 `nums` 的前 `k` 个元素包含不等于 `val` 的元素。`nums` 的其余元素和 `nums` 的大小并不重要。
返回 `k`。

>示例 2：
输入：nums = [0,1,2,2,3,0,4,2], val = 2
输出：5, nums = [0,1,4,0,3, \_ , \_ , \_]
解释：你的函数应该返回 k = 5，并且 nums 中的前五个元素为 0,0,1,3,4。
注意这五个元素可以任意顺序返回。
你在返回的 k 个元素之外留下了什么并不重要（因此它们并不计入评测）。

```python
class Solution(object):
    def removeElement(self, nums, val):
        # 还是要用到双指针，像这种不能使用额外空间的，原地修改的
        slow = 0
        for fast in range(len(nums)):
            if nums[fast] != val:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

---

# 26. 删除有序数组中的重复项
给你一个 非严格递增排列 的数组 `nums` ，请你 原地 删除重复出现的元素，使每个元素 只出现一次 ，返回删除后数组的新长度。元素的 相对顺序 应该保持 一致 。然后返回 `nums` 中唯一元素的个数。

考虑 `nums` 的唯一元素的数量为 `k` ，你需要做以下事情确保你的题解可以被通过：

更改数组 `nums` ，使 `nums` 的前 `k` 个元素包含唯一元素，并按照它们最初在 `nums` 中出现的顺序排列。`nums` 的其余元素与 `nums` 的大小不重要。
返回 `k` 。

>示例 2：
输入：nums = [0,0,1,1,1,2,2,3,3,4]
输出：5, nums = [0,1,2,3,4]
解释：函数应该返回新的长度 5 ， 并且原数组 nums 的前五个元素被修改为 0, 1, 2, 3, 4 。不需要考虑数组中超出新长度后面的元素。

```python
class Solution(object):
    def removeDuplicates(self, nums):
        i = 0
        for j in range(1,len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]
        return i+1

# OR Standard Solution
class Solution(object):
    def removeDuplicates(self, nums):
        slow = 1
        for fast in range(1, len(nums)):
            if nums[fast] != nums[fast - 1]:
                nums[slow] = nums[fast]
                slow += 1
        return slow

# OR 通用解法
class Solution(object):
    def removeDuplicates(self, nums):
        slow = 1
        for fast in range(1, len(nums)):
            if nums[fast] != nums[slow - 1]: # 永远和当前保留的最后一个比
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

---

# 80. 删除有序数组中的重复项 II
给你一个有序数组 `nums` ，请你 **原地** 删除重复出现的元素，使得出现次数超过两次的元素只出现两次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 **原地** 修改输入数组 并在使用 **O(1)** 额外空间的条件下完成。

```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 2:
            return n
        
        slow = 2
        for fast in range(2, n):
            if nums[fast] != nums[slow - 2]:
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

---

# 总结. 删除有序数组中的重复项(K)
给你一个有序数组 `nums` ，请你 **原地** 删除重复出现的元素，使得出现次数超过`K`次的元素只出现`K`次 ，返回删除后数组的新长度。

不要使用额外的数组空间，你必须在 **原地** 修改输入数组 并在使用 **O(1)** 额外空间的条件下完成。

```python
class Solution:
    def removeDuplicates(nums, k):
        if len(nums) <= k:
            return len(nums)

        slow = k
        for fast in range(k, len(nums)):
            if nums[fast] != nums[slow - k]:    #判断当前保留的边界，前k个都相等了，就不能再保留了
                nums[slow] = nums[fast]
                slow += 1
        return slow
```

---

# 169. 多数元素

给定一个大小为 `n` 的数组，找到其中的多数元素。多数元素是指在数组中出现次数 **大于** `⌊ n/2 ⌋` 的元素。

你可以假设数组是非空的，并且给定的数组总是存在多数元素。

```python
class Solution(object):
    def majorityElement(self, nums):
        counter = Counter(nums) # 这是个字典

        return max(counter.keys(), key=counter.get)   
```
|时间复杂度|空间复杂度|
|  ---    |   ---    |
|   O(n)  |  O(n)    |

#### ✅ Boyer-Moore 投票算法代码：

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        count = 0

        for num in nums:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)

        return candidate
```
| 时间复杂度 | 空间复杂度 |
|------------|------------|
| O(n) | O(1) | 

---

# 189. 轮转数组

给定一个数组，将数组中的元素向右移动 `k` 个位置，其中 `k` 是非负数。

>示例 1:
输入: nums = [1,2,3,4,5,6,7], k = 3
输出: [5,6,7,1,2,3,4]
解释:
向右轮转 1 步: [7,1,2,3,4,5,6]
向右轮转 2 步: [6,7,1,2,3,4,5]
向右轮转 3 步: [5,6,7,1,2,3,4]

```python
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        # 最优解，时间复杂度O(n)，空间复杂度O(1)
        n = len(nums)
        k %= n
        
        nums.reverse()
        nums[:k] = nums[:k][::-1]
        nums[k:] = nums[k:][::-1]
```

---

# 121. 买卖股票的最佳时机

给定一个数组 `prices`，它的第 `i` 个元素表示一支给定股票第 `i` 天的价格。

你只能选择某一天买入这只股票，并选择在未来的某一个不同的日子卖出该股票。设计一个算法来计算你所能获取的最大利润。

返回你可以从这笔交易中获取的最大利润。如果你不能获取任何利润，返回 `0` 。

注意：你不能在买入股票前卖出股票。

>示例 1：
输入：[7,1,5,3,6,4]
输出：5
解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5 。
     注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        min_price = float('inf')
        n = len(prices)

        for price in prices:
            if price < min_price:
                min_price = price
            else:
                max_profit = max(max_profit, (price - min_price))
        
        return max_profit
```
| 时间复杂度| 空间复杂度| 是否最优|
|----------------|-------------|---|
| O(n)       | O(1)       | ✅ 是最优解 |
---

# 122. 买卖股票的最佳时机 II

给定一个数组 `prices`，其中 `prices[i]` 是一支给定股票第 `i` 天的价格。

在每一天，你可能会决定购买和/或出售股票。你在任何时候 **最多** 只能持有 **一股** 股票。你也可以购买它，然后在 **同一天** 出售。设计一个算法来计算你所能获取的*最大*利润。

>示例 1：
输入：prices = [7,1,5,3,6,4]
输出：7
解释：在第 2 天（股票价格 = 1）的时候买入，在第 3 天（股票价格 = 5）的时候卖出, 这笔交易所能获得利润 = 5 - 1 = 4。
随后，在第 4 天（股票价格 = 3）的时候买入，在第 5 天（股票价格 = 6）的时候卖出, 这笔交易所能获得利润 = 6 - 3 = 3。
最大总利润为 4 + 3 = 7 。

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        profit = 0
        n = len(prices)

        for i in range(n - 1):
            if prices[i] < prices[i + 1]:
                profit += prices[i + 1] - prices[i]

        return profit
```

---

# 55. 跳跃游戏
[跳跃游戏  LeetCode链接](https://leetcode.cn/problems/jump-game/description/)
给定一个非负整数数组 `nums`，你最初位于数组的第一个下标。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

>示例 1：
输入：nums = [2,3,1,1,4]
输出：true
解释：可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。

>示例 2：
输入：nums = [3,2,1,0,4]
输出：false
解释：无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        max_reach = 0

        for i in range(n):
            if i > max_reach:
                return False
            else:
                max_reach = max(max_reach, i + nums[i])
            
        return True
```

---

# 45. 跳跃游戏 II
[跳跃游戏 II LeetCode链接](https://leetcode.cn/problems/jump-game-ii/description/)

给定一个长度为 `n` 的 0 索引整数数组 `nums`。初始位置为 `nums[0]`。
每个元素 `nums[i]` 表示从索引 `i` 向后跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:
- 0 <= j <= nums[i] 
- i + j < n
返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

>示例 1:
输入: nums = [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        n = len(nums)
        steps = 0
        end = 0
        max_reach = 0

        for i in range(n - 1):
            max_reach = max(max_reach, i + nums[i])
            if i == end:
                steps += 1
                end = max_reach
                if end >= n - 1:
                    return steps
        
        return steps

# 对于遍历，我们遍历的是能跳的位置，在终点前才能跳，所以是n - 1。
# 这道题相当于我们不直接跳，而是去遍历每一个可跳的位置，去计算它能跳的最大位置
# 如果最大位置超过当前的就更新，并且当现在的位置到达边界 end 的时候就要跳一次 step += 1。
```

---

# 274. H 指数
[H指数 LeetCode链接](https://leetcode.cn/problems/h-index/description/)

给定一位研究者的论文被引用次数的数组（`citations`），其中 `citations[i]` 表示第 `i` 篇论文被引用的次数，计算该研究者的 h 指数。

根据维基百科上 h 指数的定义：h 代表“高引用次数”，一名科研人员的 h 指数是指他（她）的 `n` 篇论文中，有 `h` 篇论文分别被引用了至少 `h` 次，其余的 `n - h` 篇论文每篇被引用次数不超过 `h` 次。

如果存在多种 h 值满足上述定义，h 指数是其中最大的那个。

>示例 1：
输入：`citations = [3,0,6,1,5]`
输出：`3` 
解释：给定数组表示研究者总共有 `5` 篇论文，每篇论文相应的被引用了 `3, 0, 6, 1, 5` 次。
     由于研究者有 `3` 篇论文每篇 至少 被引用了 `3` 次，其余两篇论文每篇被引用 不多于 `3` 次，所以她的 h 指数是 `3`。

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        citations.sort(reverse = True)
        n = len(citations)

        for i in range(n):
            if citations[i] < i + 1:
                return i

        return n
```

---

# 380. O(1) 时间插入、删除和获取随机元素
[380题 LeetCode链接](https://leetcode.cn/problems/insert-delete-getrandom-o1/description)

实现 `RandomizedSet` 类：

- `bool insert(int val)`：当元素 `val` 不存在时，向集合中插入该项，并返回 `true`；否则，返回 `false`。
- `bool remove(int val)`：当元素 `val` 存在时，从集合中移除该项，并返回 `true`；否则，返回 `false`。
- `int getRandom()`：随机返回现有集合中的一项。每个元素应该有相同的概率被返回。

你必须实现类的所有函数，并满足每个函数的 `平均` 时间复杂度为 `O(1)` 。

>示例：
输入
`["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]`
`[[], [1], [2], [2], [], [1], [2], []]`
输出
`[null, true, false, true, 2, true, false, 2]`
解释
RandomizedSet randomizedSet = new RandomizedSet();
randomizedSet.insert(1); // 向集合中插入 1 。返回 true 表示 1 被成功地插入。
randomizedSet.remove(2); // 返回 false ，表示集合中不存在 2 。
randomizedSet.insert(2); // 向集合中插入 2 。返回 true 。集合现在包含 [1,2] 。
randomizedSet.getRandom(); // getRandom 应随机返回 1 或 2 。
randomizedSet.remove(1); // 从集合中移除 1 ，返回 true 。集合现在包含 [2] 。
randomizedSet.insert(2); // 2 已在集合中，所以返回 false 。
randomizedSet.getRandom(); // 由于 2 是集合中唯一的数字，getRandom 总是返回 2 。

```python
class RandomizedSet:
    def __init__(self):
        self.arr = []
        self._dict = dict()
        # 主要的问题在于 insert() 和 remove() 如果不用字典的话就要遍历 arr 做不到O(1)
        # 所以 arr 用来存储值, _dict用来存储地址, 查询字典是O(1)的

    def insert(self, val: int) -> bool:
        if val in self._dict:
            return False
        self.arr.append(val)
        self._dict[val] = len(self.arr) - 1
        return True

    # 巧思就是为了删除这个 val, 把最后一个位置的值填充到 val 这，其实已经删除了
    # 但是要保证删除之后长度符合 -1 就把最后一个给 pop() 了
    # 这样可以保证不用去查询 val 的位置了
    # 如果不用这个，用 arr.remove(val) ，就会是 O(n) 的了
    def remove(self, val: int) -> bool:
        if val not in self._dict:
            return False
        
        idx = self._dict[val]
        last_val = self.arr[-1]

        self.arr[idx] = last_val
        self._dict[last_val] = idx

        self.arr.pop()
        del self._dict[val]
        
        return True

    def getRandom(self) -> int:
        return random.choice(self.arr)
    # random.choice()是随机生成一个数 i 然后输出 self.arr[i], 所以复杂度就是O(1)
```

---

# 238. 除自身以外数组的乘积
[238 LeetCode链接](https://leetcode.cn/problems/product-of-array-except-self)

给你一个整数数组 `nums`，返回一个数组 `answer`，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

题目要求：请不要使用除法，且在 O(n) 时间复杂度内完成此题。

>示例 1:
输入: nums = [1,2,3,4]
输出: [24,12,8,6]

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        right = left = 1

        for i in range(n):
            res[i] = left
            left *= nums[i]
        
        for j in range(n-1, -1, -1):
            res[j] *= right
            right *= nums[j]

        return res
```
- 对于这个题，涉及到前缀后缀积。一个自然的想法是对于 nums[i] 之前和之后的分别乘起来，最终 answer[i] = prefix[i] * suffix[i]。
- 如果要求`O(1)`空间复杂度，就可以直接遍历两次数组，维护 left 和 right。left是正向遍历的时候该元素左侧的乘积，赋值给 res[i]；right是反向遍历时右侧的乘积，乘给 res[i]。这样就能得到两侧乘积了。

---

# 134. 加油站
[加油站 LeetCode链接](https://leetcode.cn/problems/gas-station/description)

在一条环路上有 `n` 个加油站，其中第 `i` 个加油站有汽油 `gas[i]` 升。

你有一辆油箱容量无限的车，从第 `i` 个加油站开往第 `i+1` 个加油站需要消耗汽油 `cost[i]` 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

>示例 1:
输入: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
输出: 3
解释:
从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
因此，3 可为起始索引。

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1
        
        tank = 0    # 当前油箱容量，即cost - gas
        start = 0   # 因为要找合法的起点，所以维护start
        
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:    # 如果到不了对应的加油站就放弃这个潜在解
                            # 去看下一个节点当起点能不能行
                tank = 0
                start = i + 1

        return start
```

---

# 135. 分发糖果

[分发糖果 LeetCode链接](https://leetcode.cn/problems/candy)

有 `n` 个孩子站成一排。给你一个整数数组 `ratings` 表示每个孩子的评分。

你需要按照以下要求，给这些孩子分发糖果：

- 每个孩子至少分配到 1 个糖果。
- 相邻的孩子中，评分较高的孩子必须获得更多的糖果。

请你给每个孩子分发糖果，计算并返回需要准备的最少糖果数目。

>示例 1：
输入：ratings = [1,0,2]
输出：5
解释：你可以分别给第一个、第二个、第三个孩子分发 2、1、2 颗糖果。

>示例 2：
输入：ratings = [1,2,2]
输出：4
解释：你可以分别给第一个、第二个、第三个孩子分发 1、2、1 颗糖果。
     第三个孩子只得到 1 颗糖果，这满足题面中的两个条件。

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        n = len(ratings)
        candies = [1] * n

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candies[i] = candies[i-1] + 1
        
        for i in range(n-2, -1, -1):
            if ratings[i] > ratings[i+1] and candies[i] <= candies[i+1]:
                candies[i] = candies[i+1] + 1
        
        return sum(candies)


```
>其实这个题只需要注意两点: 
1.反向遍历的时候要注意 `candies[i] <= candies[i+1]`这个判断, 来确保本来是正确的分发结果又加了一个
2.在增加糖果的时候要注意是要比`被比较`的那一个孩子的糖果多，而不是自身加一个就行了

---

# 42. 接雨水

[接雨水 Leetcode链接](https://leetcode.cn/problems/trapping-rain-water)

示例 1：
<div align="center">
  <img src="figure/rainwatertrap.png" width="480"><br>
  <b>图 5-2 流失预测模型的SHAP特征重要性条形图</b>
</div>

> 输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：上面是由数组 [0,1,0,2,1,0,1,3,2,1,2,1] 表示的高度图，在这种情况下，可以接 6 个单位的雨水（蓝色部分表示雨水）。

示例 2：

>输入：height = [4,2,0,3,2,5]
输出：9

```python
# 动态规划
class Solution(object):
    def trap(self, height):
        n = len(height)
        if n < 3:
            return 0

        rain = 0
        left_max = [height[0]] * n
        for i in range(1,n):
            left_max[i] = max(left_max[i-1], height[i])

        right_max = [height[n-1]] * n
        for i in range(n-2,-1,-1):
            right_max[i] = max(right_max[i+1], height[i])

        for i in range(1,n-1):
            rain += min(left_max[i], right_max[i]) - height[i]
        
        return rain

# 双指针
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        left, right = 0, n-1
        left_max = right_max = 0
        res = 0
        
        while left < right:
            if height[left] < height[right]:
                if height[left] < left_max:
                    res += left_max - height[left]
                else:
                    left_max = max(left_max, height[left])
                left += 1
            else:
                if height[right] < right_max:
                    res += right_max - height[right]
                else:
                    right_max = max(right_max, height[right])
                right -= 1
        
        return res
```

对于接雨水问题的核心理解如下：

1. 在双指针解法中：
   - 每次比较 `height[left]` 和 `height[right]`：
     - 如果 `left` 比 `right` 小，说明此时的 `left` 是限制因素；
       - 如果 `height[left] < left_max`，则可接水：`left_max - height[left]`
     - 反之，`right` 是限制因素；
       - 如果 `height[right] < right_max`，则可接水：`right_max - height[right]`

2. 在动态规划解法中：
   - 预处理出每个位置的 `left_max[i]` 和 `right_max[i]`
   - 然后统一遍历，用 `min(left_max[i], right_max[i]) - height[i]` 来计算每个位置的储水量

DP 本质上就是把双指针中动态维护的最大值提前存好，最后集中计算。

---

# 13. 罗马数字转整数

[罗马数字转整数 LeetCode链接](https://leetcode.cn/problems/roman-to-integer)

罗马数字包含以下七种字符: I， V， X， L，C，D 和 M。

字符          数值
I             1
V             5
X             10
L             50
C             100
D             500
M             1000
例如， 罗马数字 2 写做 II ，即为两个并列的 1 。12 写做 XII ，即为 X + II 。 27 写做  XXVII, 即为 XX + V + II 。

通常情况下，罗马数字中小的数字在大的数字的右边。但也存在特例，例如 4 不写做 IIII，而是 IV。数字 1 在数字 5 的左边，所表示的数等于大数 5 减小数 1 得到的数值 4 。同样地，数字 9 表示为 IX。这个特殊的规则只适用于以下六种情况：

- I 可以放在 V (5) 和 X (10) 的左边，来表示 4 和 9。
- X 可以放在 L (50) 和 C (100) 的左边，来表示 40 和 90。 
- C 可以放在 D (500) 和 M (1000) 的左边，来表示 400 和 900。
给定一个罗马数字，将其转换成整数。

示例 1:

> 输入: s = "III"
输出: 3

示例 2:

>输入: s = "MCMXCIV"
输出: 1994
解释: M = 1000, CM = 900, XC = 90, IV = 4.

```python
class Solution:
    def romanToInt(self, s: str) -> int:
        roman_map = {
            'I':1,
            'V':5,
            'X':10,
            'L':50,
            'C':100,
            'D':500,
            'M':1000
        }
        res = 0

        for i in range(len(s)-1):
            if roman_map[s[i]] < roman_map[s[i+1]]:
                res -= roman_map[s[i]]
            else:
                res += roman_map[s[i]]
        
        return res + roman_map[s[-1]]
```

这个问题最值得注意的就是不要复杂化，题目要求所有的罗马数字都是合法的，也就是大的在前小的在后，小的在前的唯一情况就是那几个减法的`4,9,40,90,400,900`，找到减法只需要判断前一个和后一个的值大小就行。

---

# 12. 整数转罗马数字

[LeetCode链接](https://leetcode.cn/problems/integer-to-roman)

七个不同的符号代表罗马数字，其值如下：

|符号|值|
|---|---|
|I|	1|
|V|	5|
|X|	10|
|L|	50|
|C|	100|
|D|	500|
|M|	1000|

罗马数字是通过添加从最高到最低的小数位值的转换而形成的。将小数位值转换为罗马数字有以下规则：

- 如果该值不是以 4 或 9 开头，请选择可以从输入中减去的最大值的符号，将该符号附加到结果，减去其值，然后将其余部分转换为罗马数字。
- 如果该值以 4 或 9 开头，使用 减法形式，表示从以下符号中减去一个符号，例如 4 是 5 (V) 减 1 (I): IV ，9 是 10 (X) 减 1 (I)：IX。仅使用以下减法形式：4 (IV)，9 (IX)，40 (XL)，90 (XC)，400 (CD) 和 900 (CM)。
- 只有 10 的次方（I, X, C, M）最多可以连续附加 3 次以代表 10 的倍数。你不能多次附加 5 (V)，50 (L) 或 500 (D)。如果需要将符号附加4次，请使用 减法形式。

给定一个整数，将其转换为罗马数字。

示例 ：

>输入：num = 3749
输出： "MMMDCCXLIX"
解释：
3000 = MMM 由于 1000 (M) + 1000 (M) + 1000 (M)
 700 = DCC 由于 500 (D) + 100 (C) + 100 (C)
  40 = XL 由于 50 (L) 减 10 (X)
   9 = IX 由于 10 (X) 减 1 (I)
注意：49 不是 50 (L) 减 1 (I) 因为转换是基于小数位

```python
class Solution(object):
    def replace_rom(self, n, four, nine, five, one, roman_num):
        if n == 4:
            roman_num += four
        elif n == 9:
            roman_num += nine
        elif n == 5:
            roman_num += five
        elif n < 5:
            roman_num += n * one
        else:
            roman_num += roman3 + (n - 5) * roman4

        return roman_num


    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        roman_num = ""
        thousands = num // 1000
        hundreds = (num % 1000) // 100
        tens = (num % 100) // 10
        ones = num % 10

        roman_num += thousands * "M"

        roman_num = self.replace_rom(hundreds,'CD','CM','D','C',roman_num)
        roman_num = self.replace_rom(tens,'XL','XC','L','X',roman_num)
        roman_num = self.replace_rom(ones,'IV','IX','V','I',roman_num)
        
        return roman_num
```
```Python
# 解题思路中我觉得比较好的一个
class Solution(object):
    def intToRoman(self, num):
        roman_map = {
            'M':1000,
            'CM':900,
            'D':500,
            'CD':400,
            'C':100,
            'XC':90,
            'L':50,
            'XL':40,
            'X':10,
            'IX':9,
            'V':5,
            'IV':4,
            'I':1
        }
        res = ''

        for roman in roman_map:
            roman_num = roman_map[roman]
            if num // roman_num != 0:
                count = num // roman_num
                res += roman * count
                num %= roman_num

        return res
```

这个题好像没有特别简略的方法，主要就是要枚举各个位数可能出现的情况

---

# 58. 最后一个单词的长度

[LeetCode链接](https://leetcode.cn/problems/length-of-last-word)

给你一个字符串 s，由若干单词组成，单词前后用一些空格字符隔开。返回字符串中 最后一个 单词的长度。

单词 是指仅由字母组成、不包含任何空格字符的最大子字符串。

 

示例 1：

>输入：s = "Hello World"
输出：5
解释：最后一个单词是“World”，长度为 5。

示例 2：

>输入：s = "   fly me   to   the moon  "
输出：4
解释：最后一个单词是“moon”，长度为 4。

示例 3：

>输入：s = "luffy is still joyboy"
输出：6
解释：最后一个单词是长度为 6 的“joyboy”。

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        return len(s.split()[-1])
```

---


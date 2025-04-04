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

给你一个整数数组 `nums`，返回一个数组 `answer`，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 之外其余各元素的乘积。

题目要求：请不要使用除法，且在 O(n) 时间复杂度内完成此题。

>示例 1:
输入: nums = [1,2,3,4]
输出: [24,12,8,6]

```python
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        answer = [1] * n

        left_product = 1
        for i in range(n):
            answer[i] = left_product
            left_product *= nums[i]

        right_product = 1
        for i in range(n - 1, -1, -1):
            answer[i] *= right_product
            right_product *= nums[i]
        
        return answer
```

---

# 134. 加油站

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
class Solution(object):
    def canCompleteCircuit(self, gas, cost):
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        if sum(gas) < sum(cost):
            return -1
        
        start = 0
        cur = 0
        for i in range(len(gas)):
            cur += gas[i] - cost[i]

            if cur < 0:
                start = i + 1
                cur = 0

        return start
```

---

# 135. 分发糖果

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
class Solution(object):
    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        n = len(ratings)
        candies = [1] * n  # 每个孩子至少 1 颗糖

        # 从左到右遍历，保证右边比左边评分高时糖果数多
        for i in range(1, n):
            if ratings[i] > ratings[i - 1]:
                candies[i] = candies[i - 1] + 1

        # 从右到左遍历，保证左边比右边评分高时糖果数多
        for i in range(n - 2, -1, -1):
            if ratings[i] > ratings[i + 1] and candies[i] <= candies[i + 1]:
                candies[i] = candies[i + 1] + 1

        return sum(candies)  # 计算最少需要的糖果数
```

---


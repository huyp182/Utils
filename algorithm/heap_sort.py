def heap_sort(nums):
    def heapify(nums, low, high):
        root = low
        while True:
            child = root * 2 + 1
            if child > high:
                break
            if child < high and nums[child] < nums[child+1]:
                child += 1
            if nums[root] < nums[child]:
                nums[root], nums[child] = nums[child], nums[root]
                root = child
            else:
                break

    length = len(nums)

    # 1. 建堆
    for i in range((length-2) // 2, -1, -1):
        heapify(nums, i, length - 1)

    # 2. 出数，按升序排列
    for i in range(length - 1, 0, -1):
        nums[0], nums[i] = nums[i], nums[0]
        heapify(nums, 0, i-1)

def findKthLargest(nums, k):
    # def randomized_select(nums, begin, end, k):
    #     # print(nums, begin, end, k)
    #     if begin == end:
    #         return nums[begin]
    #     if begin < end:
    #         pivot = nums[begin]
    #         i, j = begin, end
    #         while i < j:
    #             while i < j and nums[j] < pivot:
    #                 j -= 1
    #             nums[i] = nums[j]
    #             while i < j and nums[i] >= pivot:
    #                 i += 1
    #             nums[j] = nums[i]
    #         nums[i] = pivot
    #     tmp = i
    #     if tmp < k:
    #         return randomized_select(nums, i + 1, end, k)
    #     elif tmp > k:
    #         return randomized_select(nums, begin, i - 1, k)
    #     else:
    #         return nums[k]

    # length = len(nums)
    # return randomized_select(nums, 0, length - 1, k - 1)

    def min_heapify(nums, low, high):
        root = low
        while True:
            child = root * 2 + 1
            if child > high:
                break
            if child < high and nums[child] > nums[child + 1]:
                child += 1
            if nums[root] > nums[child]:
                nums[root], nums[child] = nums[child], nums[root]
                root = child
            else:
                break

    for i in range((k - 2) // 2, -1, -1):
        min_heapify(nums, i, k - 1)

    for i in range(len(nums) - 1, k - 1, -1):
        if nums[i] > nums[0]:
            nums[0] = nums[i]
            min_heapify(nums, 0, k - 1)

    return nums[0]


if __name__ == '__main__':
    nums = [23,56,12,-9,87,3457,2124,3000,-9, 0,21323]
    heap_sort(nums)
    print(nums)

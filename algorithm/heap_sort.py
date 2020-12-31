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


if __name__ == '__main__':
    nums = [23,56,12,-9,87,3457,2124,3000,-0,21323]
    heap_sort(nums)
    print(nums)

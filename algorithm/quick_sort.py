def quick_sort(nums, begin, end):
    if begin < end:
        pivot = nums[begin]
        i, j = begin, end
        while i < j:
            while i < j and nums[j] > pivot:
                j -= 1
            nums[i] = nums[j]
            while i < j and nums[i] <= pivot:
                i += 1
            nums[j] = nums[i]
        nums[i] = pivot
        quick_sort(nums, begin, i - 1)
        quick_sort(nums, i + 1, end)


if __name__ == '__main__':
    nums = [2,3,1,1,9]
    quick_sort(nums, 0, len(nums) - 1)
    print(nums)

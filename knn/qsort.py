import numpy as np

def qsort(a, left, right):

	"""
	quick sort method
	"""
	# Finish the sort routine
	if left >= right:
		return 
	i = left
	j = right
	key = a[left]
	while i < j:
		while i < j and key <= a[j]:
			j -= 1

		a[i] = a[j]

		while i < j and key >= a[i]:
			i += 1

		a[j] = a[i]

	a[i] = key
	qsort(a, left, i - 1)
	qsort(a, i + 1, right)


if __name__ == "__main__":

	arr = list(range(100))
	np.random.shuffle(arr)

	qsort(arr, 0, len(arr) - 1)

	print(arr)

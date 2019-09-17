import numpy as np

def partition(a, l, r):

	key = a[r]
	i = l

	for j in range(l, r):
		if a[j] <= key:
			a[i], a[j] = a[j], a[i]
			i += 1

	a[i], a[r] = a[r], a[i]
	return i


def qselect(a, l, r, k):

	if k > 0 and k <= r - l + 1:

		index = partition(a, l, r)

		if index - l == k - 1:
			return a[index]

		if index - l > k - 1
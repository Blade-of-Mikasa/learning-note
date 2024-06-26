# 竞赛常用STL

## 目录

#### 函数

* sort 函数
* lower_bound & upper_bound 函数

#### 容器

* vector 向量
* map 映射
* set 集合
* multiset 多重集合
* queue 队列
* deque 双向队列
* priority_queue 优先队列
* stack 栈
* link 双向链表

## 函数

### sort 函数

时间复杂度为 $O(nlogn)$的排序函数

使用模板：

```
sort(begin,end,cmp());
```

其中，begin，end 是目标容器的地址；

cmp 函数是排序方式，没有 cmp 函数时，默认为升序。

对于大部分情况，需要自定义 cmp（如下，实现了降序排列）

```
bool cmp(int a,int b)
{
	return a<b;
}
```

### lower_bound & upper_bound 函数

lower_bound & upper_bound ：在一个有序且支持随机的容器中，查找第一个大于等于/大于给定值的元素的位置。如果容器中存在相等的元素，则返回第一个相等元素的位置。

模板

```text
lower_bound(begin, end, cmp());
```

## 容器

### vector 向量

`vector` 是 C++ 中的动态数组容器，它提供了一种能够在运行时调整大小的数组。`vector` 中的元素是连续存储的，并且可以通过索引进行快速访问。

##### 模板

```
vector<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查向量是否为空。如果向量为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回向量中元素的数量。时间复杂度为 O(1)。
- `.push_back(element)`：将元素添加到向量的末尾。如果向量的容量不足，会自动进行扩容操作。时间复杂度为 O(1)（平摊）。
- `.pop_back()`：删除向量末尾的元素。时间复杂度为 O(1)。
- `.insert(position, element)`：在指定位置之前插入元素。时间复杂度为 O(n)，其中 n 是向量的大小。
- `.erase(position)`：删除指定位置的元素。时间复杂度为 O(n)，其中 n 是向量的大小。
- `.erase(start, end)`：删除指定范围内的元素。时间复杂度为 O(n)，其中 n 是要删除的元素数量。
- `[]` 操作符：通过索引访问向量中的元素。时间复杂度为 O(1)。

### map 映像

`map` 是 C++ 中的关联容器，它提供了一种键值对的映射关系。`map` 中的每个元素都包含一个键和一个值，可以通过键来访问对应的值。以下是关于 `std::map` 的一些重要信息：

##### 模板

```
map<key_type, value_type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查 map 是否为空。时间复杂度为 O(1)。
- `.size()`：返回 map 中键值对的数量。时间复杂度为 O(1)。
- `.insert(pair<key_type, value_type>(key, value))`：插入一个键值对到 map 中。时间复杂度为 O(log n)，其中 n 是 map 中键值对的数量。
- `.erase(key)`：删除 map 中指定键的键值对。时间复杂度为 O(log n)，其中 n 是 map 中键值对的数量。
- `.find(key)`：返回一个迭代器，指向 map 中具有指定键的元素，如果找不到指定键，则返回指向 map 结尾的迭代器。时间复杂度为 O(log n)，其中 n 是 map 中键值对的数量。

### set 集合

`set` 是 C++ 中的关联容器，它提供了一种有序且不重复的集合。`set` 中的每个元素都是唯一的，并且按照特定的排序准则进行排序。

```
set<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查 set 是否为空。如果 set 为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回 set 中元素的数量。时间复杂度为 O(1)。
- `.insert(element)`：将元素插入到 set 中。时间复杂度为 O(log n)，其中 n 是 set 中元素的数量。
- `.erase(element)`：删除 set 中指定的元素。时间复杂度为 O(log n)，其中 n 是 set 中元素的数量。
- `.find(element)`：返回一个迭代器，指向 set 中具有指定值的元素，如果找不到指定值，则返回指向 set 结尾的迭代器。时间复杂度为 O(log n)，其中 n 是 set 中元素的数量。

### multiset 多重集合

与 `set` 不同，`multiset` 允许存储重复的元素。

##### 模板

```
mulitset<type> name;
```

##### 相关函数和时间复杂度

- `.count(value)`：返回在 `std::multiset` 中等于给定值的元素的数量。
- 其他同 set

### queue 队列

`queue` 是 C++ 中的队列容器，它遵循先进先出（FIFO）的原则。在队列中，新元素总是被插入到末尾，而最早插入的元素总是被删除。

##### 模板

```
queue<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查队列是否为空。如果队列为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回队列中元素的数量。时间复杂度为 O(1)。
- `.push(element)`：将元素添加到队列的末尾。时间复杂度为 O(1)。
- `.pop()`：删除队列开头的元素。时间复杂度为 O(1)。
- `.front()`：返回队列开头的元素的引用。时间复杂度为 O(1)。
- `.back()`：返回队列末尾的元素的引用。时间复杂度为 O(1)。

### deque 双向队列

`deque` 是 C++ 中的一个双端队列容器，它遵循先进先出（FIFO）的原则。在队列中，新元素总是被插入到末尾，而最早插入的元素总是被删除。

##### 模板

```
deque<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查双端队列是否为空。如果双端队列为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回双端队列中元素的数量。时间复杂度为 O(1)。
- `.push_back(element)`：将元素添加到双端队列的末尾。时间复杂度为 O(1)。
- `.push_front(element)`：将元素添加到双端队列的开头。时间复杂度为 O(1)。
- `.pop_back()`：删除双端队列末尾的元素。时间复杂度为 O(1)。
- `.pop_front()`：删除双端队列开头的元素。时间复杂度为 O(1)。
- `.front()`：返回双端队列开头的元素的引用。时间复杂度为 O(1)。
- `.back()`：返回双端队列末尾的元素的引用。时间复杂度为 O(1)。

### priority_queue 优先队列

`priority_queue` 是 C++ 中的一个容器，它按照元素的优先级进行排序和访问。默认情况下，它是一个最大堆（大顶堆），即优先级高的元素会被放置在队列的前面。

##### 模板

```
priority_queue<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查优先队列是否为空。如果优先队列为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回优先队列中元素的数量。时间复杂度为 O(1)。
- `.push(element)`：将元素添加到优先队列中。时间复杂度为 O(log n)，其中 n 是当前优先队列的大小。
- `.pop()`：删除优先队列中的顶部元素（优先级最高的元素）。时间复杂度为 O(log n)，其中 n 是当前优先队列的大小。
- `.top()`：返回优先队列中顶部元素（优先级最高的元素）的引用。时间复杂度为 O(1)。

### stack 栈

栈是一种具有后进先出（Last-In-First-Out，LIFO）特性的数据结构。在栈中，元素按照插入的顺序排列，并且只能在栈顶插入和删除元素。

##### 功能

- 栈的主要功能是用于存储和管理元素，按照后进先出的顺序进行操作。新元素总是被插入到栈顶，而最后插入的元素总是最先被删除。

##### 模板

```
stack<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查栈是否为空。如果栈为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回栈中元素的数量。时间复杂度为 O(1)。
- `.top()`：返回栈顶元素的引用。如果栈为空，则行为是未定义的。时间复杂度为 O(1)。
- `.push(element)`：将元素插入到栈顶。时间复杂度为 O(1)。
- `.pop()`：删除栈顶元素。时间复杂度为 O(1)。

### link 链表

`list` 是 C++ 中的双向链表（doubly linked list）容器，它提供了一种高效地插入和删除元素的方式。与向量（`vector`）和数组（`array`）不同，`list` 在任意位置插入和删除元素的开销是常数时间。

##### 模板

```
list<type> name;
```

##### 相关函数和时间复杂度

- `.empty()`：检查链表是否为空。如果链表为空，则返回 `true`；否则返回 `false`。时间复杂度为 O(1)。
- `.size()`：返回链表中元素的数量。时间复杂度为 O(1)。
- `.push_back(element)`：将元素插入到链表的末尾。时间复杂度为 O(1)。
- `.push_front(element)`：将元素插入到链表的开头。时间复杂度为 O(1)。
- `.pop_back()`：删除链表末尾的元素。时间复杂度为 O(1)。
- `.pop_front()`：删除链表开头的元素。时间复杂度为 O(1)。
- `.insert(position, element)`：在指定位置之前插入元素。时间复杂度为 O(1)。
- `.erase(position)`：删除指定位置的元素。时间复杂度为 O(1)。
- `.remove(element)`：删除链表中所有与指定值相等的元素。时间复杂度为 O(n)，其中 n 是链表的大小。




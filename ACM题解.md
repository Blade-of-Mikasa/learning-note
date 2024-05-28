# ACM题解

### Part1.1动态规划

#### 前言

[告别动态规划，连刷40道动规算法题，我总结了动规的套路-CSDN博客](https://blog.csdn.net/hollis_chuang/article/details/103045322?spm=1001.2014.3001.5506)

**第一步骤**：定义**数组元素的含义**。我们会用一个数组，来保存历史数组，假设用一维数组 dp[] 吧。这个时候有一个非常非常重要的点，就是规定你这个数组元素的含义，例如你的 dp[i] 是代表什么意思？

**第二步骤**：找出**数组元素之间的关系式**。我觉得动态规划，还是有一点类似于我们高中学习时的**归纳法**的，当我们要计算 dp[n] 时，是可以利用 dp[n-1]，dp[n-2]…..dp[1]，来推出 dp[n] 的，也就是可以利用**历史数据**来推出新的元素值，所以我们要找出数组元素之间的关系式，例如 dp[n] = dp[n-1] + dp[n-2]，这个就是他们的关系式了。而这一步，也是最难的一步，后面我会讲几种类型的题来说。

**第三步骤**：找出**初始值**。学过**数学归纳法**的都知道，虽然我们知道了数组元素之间的关系式，例如 dp[n] = dp[n-1] + dp[n-2]，我们可以通过 dp[n-1] 和 dp[n-2] 来计算 dp[n]，但是，我们得知道初始值啊，例如一直推下去的话，会由 dp[3] = dp[2] + dp[1]。而 dp[2] 和 dp[1] 是不能再分解的了，所以我们必须要能够直接获得 dp[2] 和 dp[1] 的值，而这，就是**所谓的初始值**。

#### 数字三角形

[P1216 [USACO1.5\] [IOI1994]数字三角形 Number Triangles - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1216)

```c++
#define maxn 105
int a[maxn][maxn],dp[maxn][maxn];

for(int i = 1; i <= n; ++ i)
        dp[n][i] = a[n][i];

for(int i = n - 1; i >= 1; -- i)
    for(int j = 1; j <= i; ++ j)
        dp[i][j] = max(a[i][j] + dp[i + 1][j], a[i][j] + dp[i + 1][j + 1]);
```

#### 01背包问题

[P1048 [NOIP2005 普及组\] 采药 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1048)

```c++
#define maxn 105
int dp[maxn][1005],t[maxn],v[maxn];

for(int i = 1; i <= M; ++ i)
    for(int j = 1; j <= T; ++ j)
    {
        if(j >= t[i])
            dp[i][j] = max(dp[i-1][j],dp[i-1][j - t[i]] + v[i])
        else
            dp[i][j] = dp[i-1][j];
    }
```

#### 01背包问题（变式）

[P1049 [NOIP2001 普及组\] 装箱问题 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1049)

**注意到$ w_i=v_i$**

```c++
#define MAXN 35//数组大小
int dp[MAXN][20005], v[MAXN];

dp[0][V] = 0;

for(int i = 1; i <= n; ++ i)
    for(int j = 1; j <= V; ++ j)
    {
        if(j >= v[i])
            dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - v[i]] + v[i]);
        else
            dp[i][j] = dp[i - 1][j];
    }
int res = -INF;
for(int j = 0; j <= V; ++ j)
    res = max(dp[n][j], res);
cout<<V - res<<endl;

```

#### 完全背包问题

[P1616 疯狂的采药 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1616)

优化后的dp，降低时间、空间复杂度

```c++
for(int i = 1; i <= m; ++ i)
	for(int j = 1; j <= t; ++ j)
    {
		int maxk = j / a[i];
        for(int k = 0; k <= maxk; ++ k)
        {
            dp[j] = max(dp[j], dp[j - k * a[i]] + k * b[i]);
        }
    }
```

时间复杂度$o(n^3)$（来自 oi wiki）

学长给出的时间优化：

```c++
for(int i = 1; i <= m; ++ i)
	for(int j = a[i]; j <= t; ++ j)
        dp[j] = max(dp[j], dp[j - a[i]] + b[i]);
```

时间复杂度$o(n^2)$

#### 可选性背包

[Problem - E - Codeforces](https://codeforces.com/contest/1974/problem/E)

我想到的是反悔贪心（边走边拿，能拿下就把拿到的记录到某个数据结构 A 中；拿不下就反悔，从 A 中用 01背包选反悔的）

实际上就是给状态转移加个判断，满足就可以转移，反之则不能。

对于该题，还要注意：考虑到价值和体积的大小，背包的第二参量应为价值，即 $dp_{ij}$ 表示"当遍历到第 i 个物品、得到 j 价值时，花费体积的最小值"

故有：
$$
if(dp[i - 1][j] + w[i] <= (i - 1) * x )\\
dp[i][j] = min(dp[i - 1][j], dp[i - 1][j + v[i]] + w[i])\\
otherwise\\
dp[i][j] = dp[i - 1][j]\\
$$
 

#### 改变参量的背包

#### 最大的连续子段和

[P1115 最大子段和 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1115)

自己做的时候只想出前缀和+穷举，复杂度$O ( n ^ 2 )$.

不要思维定式地用前缀和，nor 潜意识地认为子段和就比单项大。DP的关键在于递推式，所以重要的是**找到局部的递推过程**！

```c++
for(int i = 1; i <= n; ++i)
    dp[i] = max(dp[i-1] + s[i], s[i]);
```

#### 等跨度路线（特定点不可取）

[P1002 [NOIP2002 普及组\] 过河卒 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1002)

1、抵消 -1 标记

2、没开long long ，结果溢出（2024.5.22 以后，已经把 int 声明为 long long 的宏了，这个问题大概不会再出现了）

```c++
int a[MAXN][MAXN];
int dx[] = {-2, -2, -1, -1, 1, 1, 2, 2, 0};
int dy[] = {1, -1, 2, -2, 2, -2, 1, -1, 0};


for(int i = 0; i < 9; ++ i)
{
    if(x + dx[i] >= 0 && y + dy[i] >= 0)
    a[x + dx[i]][y + dy[i]] = -1;
}

for(int i = 1; i <= n; ++i)
{
    if(a[i][0] == -1)
        break;
    a[i][0] = 1;
}
for(int j = 1; j <= m; ++j)
{
    if(a[0][j] == -1)
        break;
    a[0][j] = 1;
}

for(int i = 1; i <= n; ++ i)
	for(int j = 1; j <= m; ++ j)
    {
        if(a[i][j] != -1)
        {
            a[i][j] = a[i - 1][j] + a[i][j - 1];
            //抵消-1
            if(a[i - 1][j] == -1)
                ++ a[i][j];
            if(a[i][j - 1] == -1)
                ++ a[i][j];
        }
        else
            a[i][j] = 0;
    }

cout<<a[n][m];
```

#### 不等跨度的选路

[P1164 小A点菜 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1164)

```c++
for(int i = 0; i <= n; ++ i)
    dp[i][0] = 1;

for(int i = 1; i <= n; ++ i)
    for(int j = 1; j <= m; ++ j)
    {
        if(j >= a[i])
            dp[i][j] = dp[i - 1][j - a[i]] + dp[i - 1][j];
        else
            dp[i][j] = dp[i - 1][j];
    }
```

### Part1.2贪心

#### 最大的拼接字符串

[P1012 [NOIP1998 提高组\] 拼数 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1012)

不是单纯地比较$S_1, S_2$的字典序（反例：321 32）

应该比较$S_1+S_2,S_2+S_1$的字典序

```c++
string a[MAXN], tmp[MAXN];

void merge_sort(string a[], int l, int r)
{
    if(l >= r)return ;

    int mid = l + r >> 1;
    merge_sort(a, l, mid);
    merge_sort(a, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while(i <= mid && j <= r)
    { 
        if(a[j] + a[i] < a[i] + a[j]) tmp[k ++] = a[i ++];
        else    tmp[k ++] = a[j ++];
    }

    while(i <= mid) tmp[k ++] = a[i ++];
    while(j <= r)   tmp[k ++] = a[j ++];

    for(i = l, j = 0; i <= r; ++ i, ++ j)
        a[i] = tmp[j];
}

void solve()
{
    int n;
    cin>>n;
    for(int i = 0; i < n; ++ i)
        cin>>a[i];
    merge_sort(a, 0, n - 1);
    for(int i = 0; i < n; ++ i)
        cout<<a[i];
}
```

#### 最长的非负前缀和子序列

[问题 - C2 - Codeforces](https://codeforces.com/contest/1526/problem/C2)

遍历时，每喝一个负值药水，就把它放入优先队列（按扣了多少血）中，之后遇到喝不了的负值药水时，看看能不能反悔之前的药水。

```c++
priority_queue<int> p;
int n;
cin>>n;
for(int i = 1; i <= n; ++ i)
    cin>>a[i];

p.push(0);//防止top时引发re
int now = 0, ans = 0;
for(int i = 1; i <= n; ++ i)
{
    if(now + a[i] >= 0)
    {
        now += a[i];
        ans ++;
        if(a[i] < 0)
            p.push(- a[i]);
    }
    else if(p.top() + a[i] > 0)
    {
        now += (p.top() + a[i]);
        p.pop();
        p.push(- a[i]);
    }
}
cout<<ans;
```



### Part2.训练题

[CUGBACM23级春季学期训练#21A - Virtual Judge (vjudge.net)](https://vjudge.net/contest/627077#problem/C)

$n<500$,  给出数组$x_i<500\ (i=2,3,...n)$,  找出数组$a_i<1e9\ (i=1,2,3,...n)$,  使得$x_i=a_i\ mod\ a_{i-1}$.


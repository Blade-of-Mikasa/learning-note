# ACM题解





## 0. 基础算法

#### 前缀 & 宏

```c++
//1 MB == 1e6 int
#include <bits/stdc++.h>
#define int long long
#define FILESTREAM {freopen("input.txt","r",stdin);freopen("output.txt","w",stdout);}//输入输出流重定向（记得改文件名
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define ARRAY2D(a)  {int sizeof1 = D1, sizeof2 = D2;for(int i = BEGIN; i <= sizeof1; ++ i){cout<<"i = "<<i<<"   ";for(int j = BEGIN; j <= sizeof2; ++ j)cout<<std::right<<std::setw(3)<<a[i][j]<<' ';cout<<endl;}}//格式化输出二维数组
#define ARRAY1D(a)  {int sizeof1 = D1;for(int i = BEGIN; i <= sizeof1; ++ i)cout<<std::right<<std::setw(3)<<a[i]<<' ';cout<<endl;}//格式化输出一维数组
#define FOR(r1, r2) for(int i = 1; i <= r1; ++ i) for(int j = 1; j <= r2; ++ j)//三角遍历用FOR2D(r1, i)
#define clear(a) {memset(a, 0, sizeof(a));}
#define undirectedEdge(u, v, w) {e[u].push_back(edge(u, v, w)); e[v].push_back(edge(v, u, w));}//双向连边
using namespace std;//上面的都别动
#define BEGIN 1//数组遍历起点
#define D1 cnt//一维大小
#define D2 size - 1 //二维大小
const double pi = acos(-1.0);
const double eps = 1e-8;//精度
const int INF = 0x3f3f3f3f3f3f3f3f;
const int inf = 0x3f3f3f3f;
const int MAXN = 1e6 + 10;
const int maxn = 2e5 + 10;
//#define endl '/n'//注意该宏也作用于check（无法刷新输出缓存区

void solve(){}

signed main(){
    ISO;
    int t = 1;
    //cin>>t;
    while(t--)
        solve();
    return 0;
}
```

排序算法

**Quick Sort（快速排序）**

```c++
void quick_sort(int a[], int l, int r)
{
	if(l >= r)	return;
	
	int x = a[l], i = l - 1, j = r + 1;
	while(i < j)
	{
		do ++ i ; while(a[i] < x);
		do -- j ; while(a[j] > x);
        if(i < j)	swap(a[i], a[j]);
	}
    
	quick_sort(a, l, j);
	quick_sort(a, j + 1, r);
	return;
}
```

**Merge Sort（归并排序）**

```c++
void merge_sort(int a[], int l, int r)
{
    if(l >= r)  return;

    int mid = l + r >> 1;

    merge_sort(a, l, mid);
    merge_sort(a, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while(i <= mid && j <= r)
        if(a[i] < a[j]) tmp[k ++] = a[i ++];
        else	tmp[k ++] = a[j ++];

    while(i <= mid) tmp[k ++] = a[i ++];
    while(j <= r)   tmp[k ++] = a[j ++];

    for(i = l, j = 0; i <= r; i ++, j ++)   
        a[i] = tmp[j];
}
```

#### Binary Search（二分查找）

**整数二分**

先按题意分为两个区间（记为 A 区间和 B 区间）

```c++
// A区域的右端点。区间[l, r]被划分成[l, mid - 1]和[mid, r]时使用：
while (l < r)
{
	int mid = l + r + 1 >> 1;
	if (mid belong A) l = mid;
	else r = mid - 1;
}
//return l;
```

```c++
// B区域的左端点。区间[l, r]被划分成[l, mid]和[mid + 1, r]时使用：
while (l < r)
{
	int mid = l + r >> 1;
	if (mid belong A) l = mid + 1;
	else  r = mid;
}
//return l;
```

**lower/upper_bound**

`lower_bound` 返回第一个大于等于 value 的地址

`upper_bound` 返回第一个大于 value 的地址



**浮点数二分**

一样分为 A 区间和 B 区间

```c++
// eps 表示精度，取决于题目对精度的要求，通常要比题目要求高两个数量级
const double eps = 1e-6;   

while (r - l > eps)
{
	double mid = (l + r) / 2;
	if (mid belong A) l = mid;
	else r = mid;
}
//return l;
```

#### 三分

本质还是二分，只是判断区间 A、B 时用了两个参数

**浮点数三分**

```Cpp
while(r - l > eps)
{
    double k = (r - l) / 3.0;
    double mid1 = l + k, mid2 = r - k;
    if( f(mid1) > f(mid2) ) r = mid2;
    else    l = mid1;
}
```

**实数三分**

```cpp
while(r - l > 2)
{
    int mid1 = l + (r - l) / 3;
    int mid2 = r - (r - l) / 3;
    if( check(mid1, mid2) )
        move r;
     else
        move l;
}
```



#### High Accuracy（高精度）

**转化 & 输出**

数字转字符串

`s = to_string(n)`

字符串转高精度

```c++
for(int i = s.size() - 1; i >= 0; ++ i)
    a.push_back(s[i] - '0');
```

高精度输出

```c++
for(int i = c.size() - 1; i >= 0; -- i)
    cout<<c[i];
```



**加减乘除模**

加法

```c++
vector<int> add(vector<int> & a, vector<int> & b)
{
    vector<int> c;
    int t = 0;//进位
    for(int i = 0; i < a.size() || i < b.size(); ++ i)
    {
        if(i < a.size())	t += a[i];
        if(i < b.size())    t += b[i];
        c.push_back(t % 10);
        t /= 10;
    }
    
    if(t)   c.push_back(1);

    return c;
}
```

减法

乘法（高乘高）

```c++
vector<int> mul(vector<int> & A, int b)
{
    vector<int> C;
    int t = 0;
    for (int i = 0; i < A.size() || t; i ++ )
    {
        if (i < A.size()) t += A[i] * b;
        C.push_back(t % 10);
        t /= 10;
    }

    return C;
}
```



**除法 & 取模（高除以低）**

```c++
// A / b = C ... r, A >= 0, b > 0
vector<int> div(vector<int> &A, int b, int &r)
{
    vector<int> C;
    r = 0;
    for (int i = A.size() - 1; i >= 0; i -- )
    {
        r = r * 10 + A[i];
        C.push_back(r / b);
        r %= b;
    }
    reverse(C.begin(), C.end());
    while (C.size() > 1 && C.back() == 0) C.pop_back();
    return C;
}
```

**封装的 big_int 类**

没有负数判定

```c++
class big_int {
private:
    std::vector<short int> digits; // 用于存储大整数的每一位

    // 去除前导零
    void removeLeadingZeros() {
        while (digits.size() > 1 && digits.back() == 0) {
            digits.pop_back();
        }
    }

    // 比较两个big_int，返回-1如果*this小于other，0如果等于，1如果大于
    int compare(const big_int& other) const {
        if (digits.size() < other.digits.size()) return -1;
        if (digits.size() > other.digits.size()) return 1;
        for (int i = digits.size() - 1; i >= 0; --i) {
            if (digits[i] < other.digits[i]) return -1;
            if (digits[i] > other.digits[i]) return 1;
        }
        return 0;
    }

public:
    // 默认构造函数
    big_int() : digits(1, 0) {}

    // 从字符串构造大整数
    big_int(const std::string& number) {
        for (auto it = number.rbegin(); it != number.rend(); ++it) {
            if (!isdigit(*it)) {
                throw std::invalid_argument("Invalid character in string");
            }
            digits.push_back(*it - '0');
        }
        removeLeadingZeros();
    }

    // 从整数构造大整数
    big_int(long long number) {
        if (number == 0) {
            digits.push_back(0);
        } else {
            while (number > 0) {
                digits.push_back(number % 10);
                number /= 10;
            }
        }
    }

    // 转换为字符串
    std::string to_string() const {
        std::string result;
        for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
            result += std::to_string(*it);
        }
        return result;
    }

    // 获取有效位数
    size_t size() const {
        return digits.size();
    }

    // 加法运算符重载
    big_int operator+(const big_int& other) const {
        big_int result;
        result.digits.clear(); // 清空结果向量

        int carry = 0;
        size_t maxSize = std::max(digits.size(), other.digits.size());

        for (size_t i = 0; i < maxSize || carry; ++i) {
            int digitSum = carry;
            if (i < digits.size()) digitSum += digits[i];
            if (i < other.digits.size()) digitSum += other.digits[i];

            result.digits.push_back(digitSum % 10);
            carry = digitSum / 10;
        }

        return result;
    }

    // 加等运算符重载
    big_int& operator+=(const big_int& other) {
        *this = *this + other;
        return *this;
    }

    // 减法运算符重载
    big_int operator-(const big_int& other) const {
        if (*this < other) {
            throw std::invalid_argument("Result of subtraction cannot be negative.");
        }

        big_int result;
        result.digits.clear();

        int carry = 0;

        for (size_t i = 0; i < digits.size(); ++i) {
            int digitSub = digits[i] - carry;
            if (i < other.digits.size()) {
                digitSub -= other.digits[i];
            }

            if (digitSub < 0) {
                digitSub += 10;
                carry = 1;
            } else {
                carry = 0;
            }

            result.digits.push_back(digitSub);
        }

        result.removeLeadingZeros();
        return result;
    }

    // 减等运算符重载
    big_int& operator-=(const big_int& other) {
        *this = *this - other;
        return *this;
    }

    // 乘法运算符重载
    big_int operator*(const big_int& other) const {
        big_int result;
        result.digits.assign(digits.size() + other.digits.size(), 0);

        for (size_t i = 0; i < digits.size(); ++i) {
            int carry = 0;
            for (size_t j = 0; j < other.digits.size() || carry; ++j) {
                long long current = result.digits[i + j] +
                                    digits[i] * (j < other.digits.size() ? other.digits[j] : 0) +
                                    carry;
                result.digits[i + j] = current % 10;
                carry = current / 10;
            }
        }

        result.removeLeadingZeros();
        return result;
    }

    // 乘等运算符重载
    big_int& operator*=(const big_int& other) {
        *this = *this * other;
        return *this;
    }

    // 除法运算符重载
    big_int operator/(const big_int& other) const {
        if (other == big_int(0)) {
            throw std::invalid_argument("Division by zero");
        }

        big_int dividend(*this);
        big_int divisor(other);
        big_int quotient;

        quotient.digits.resize(dividend.digits.size());

        big_int current;

        for (int i = dividend.digits.size() - 1; i >= 0; --i) {
            current.digits.insert(current.digits.begin(), dividend.digits[i]);
            current.removeLeadingZeros();

            int x = 0;
            int l = 0, r = 10;

            while (l <= r) {
                int mid = (l + r) / 2;
                big_int temp = divisor * big_int(mid);

                if (temp <= current) {
                    x = mid;
                    l = mid + 1;
                } else {
                    r = mid - 1;
                }
            }

            quotient.digits[i] = x;
            current = current - divisor * big_int(x);
        }

        quotient.removeLeadingZeros();
        return quotient;
    }

    // 除等运算符重载
    big_int& operator/=(const big_int& other) {
        *this = *this / other;
        return *this;
    }

    // 取模运算符重载
    big_int operator%(const big_int& other) const {
        if (other == big_int(0)) {
            throw std::invalid_argument("Modulo by zero");
        }

        big_int result = *this - (*this / other) * other;
        return result;
    }

    // 取模等运算符重载
    big_int& operator%=(const big_int& other) {
        *this = *this % other;
        return *this;
    }

    // 等于运算符重载
    bool operator==(const big_int& other) const {
        return digits == other.digits;
    }

    // 不等于运算符重载
    bool operator!=(const big_int& other) const {
        return !(*this == other);
    }

    // 小于运算符重载
    bool operator<(const big_int& other) const {
        return compare(other) == -1;
    }

    // 大于运算符重载
    bool operator>(const big_int& other) const {
        return compare(other) == 1;
    }

    // 小于等于运算符重载
    bool operator<=(const big_int& other) const {
        return compare(other) <= 0;
    }

    // 大于等于运算符重载
    bool operator>=(const big_int& other) const {
        return compare(other) >= 0;
    }

    // 输入运算符重载
    friend std::istream& operator>>(std::istream& in, big_int& number) {
        std::string input;
        in >> input;
        number = big_int(input);
        return in;
    }

    // 输出运算符重载
    friend std::ostream& operator<<(std::ostream& out, const big_int& number) {
        out << number.to_string();
        return out;
    }
};
```



#### 前缀和/差分

前缀和与差分可以看做逆运算，但是处理手段略有不同

前缀和

容斥原理

$O(1)$完成静态数组的区间查询

```c++
//一维前缀和
S[i]  = s[i - 1] + a[i];//= a[1] + a[2] + ... + a[i]
a[l] + ... + a[r] = S[r] - S[l - 1]
//二维前缀和
//S[i, j] = 第i行j列格子左上部分所有元素的和; 以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵的和为：
s[i][j] = s[i][j - 1] + s[i + 1][j] - s[i -1][j -1] + a[i][j];
S[x2, y2] - S[x1 - 1, y2] - S[x2, y1 - 1] + S[x1 - 1, y1 - 1];
```

差分

$O(1)$完成动态数组的区间修改

```c++
//一维差分 —— 给区间[l, r]中的每个数加上c：
d[l] += c, d[r + 1] -= c
//二维差分 ——  差分矩阵，给以(x1, y1)为左上角，(x2, y2)为右下角的子矩阵中的所有元素加上c：
d[x1, y1] += c, d[x2 + 1, y1] -= c, d[x1, y2 + 1] -= c, d[x2 + 1, y2 + 1] += c
```

#### 离散化



```c++
int v[MAXN];//原数据
int a[MAXN], s[MAXN];

void solve(){
    int n, q;
    cin>>n;
    for(int i = 1; i <= n; ++ i)
    {
        cin>>v[i];
    }

    for(int i = 1; i <= n; ++ i)
    {
        cin>>a[i];
        s[i] = a[i] + s[i - 1];
    }
    v[n + 1] = INF;
    s[n + 1] = s[n];
    cin>>q;
    for(int i = 1; i <= q; ++ i)
    {
        int lv, rv;
        cin>>lv>>rv;
        int l = (lower_bound(v, v + n + 1, lv) - v);
        int r = (upper_bound(v, v + n + 1, rv) - v - 1);

        cout<<s[r] - s[l - 1]<<endl;
    }
}
```



#### 位运算

`lowbit(i)` or `i & -i`	返回 i 的最后一位 1

`n >> k & 1`	求 n 的第 k 位数字

`x | ( 1 << k )`	将 x 第 k 位变为1

`x ^ ( 1 << k )`	将 x 第 k 位取反

`x & ( x - 1 )`	将 x 最右边的 1 置为 0 (去掉最右边的 1 )

`x | ( x + 1 )`	将 x 最右边的 0 置为 1

`x & 1`	判断奇偶性：真为奇，假为偶

#### 双指针

**尺取法**

```c++
//check: 是否满足要求
for(int i = 1, j = 0; i <= n; ++ i)
{
    while(j < n && !check(now))
    {
        ++ j;
        //deal j
    }
    
    //if(...)
    //update ans

    //deal i
}
```



[P1638 逛画展 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1638)

```c++
//本题中用 cnt 来判断是否完成所有作品, 避免了遍历数组t, 值得学习
int a[maxn], t[maxn];

void solve(){
    int n, m;
    cin>>n>>m;
    for(int i = 1; i <= n; ++ i)
        cin>>a[i];

    int cnt = 0;
    int l, r, mina = INF;
    for(int i = 1, j = 0; i <= n; ++ i)
    {
        while(j < n && cnt < m)
        {
            ++ j;
            if(t[ a[j] ] == 0)
                cnt ++;
            t[ a[j] ] ++;
        }

        if(cnt >= m && j - i + 1 < mina)
        {
            mina = j - i + 1;
            l = i, r = j;
        }

        if(t[ a[i] ] == 1)
            cnt --;
        t[ a[i] ] --;
    }
    cout<<l<<' '<<r<<endl;
}
```



## 1. 动态规划

### 前言

[告别动态规划，连刷40道动规算法题，我总结了动规的套路-CSDN博客](https://blog.csdn.net/hollis_chuang/article/details/103045322?spm=1001.2014.3001.5506)

**第一步骤**：定义**数组元素的含义**。我们会用一个数组，来保存历史数组，假设用一维数组 dp[] 吧。这个时候有一个非常非常重要的点，就是规定你这个数组元素的含义，例如你的 dp[i] 是代表什么意思？

**第二步骤**：找出**数组元素之间的关系式**。我觉得动态规划，还是有一点类似于我们高中学习时的**归纳法**的，当我们要计算 dp[n] 时，是可以利用 dp[n-1]，dp[n-2]…..dp[1]，来推出 dp[n] 的，也就是可以利用**历史数据**来推出新的元素值，所以我们要找出数组元素之间的关系式，例如 dp[n] = dp[n-1] + dp[n-2]，这个就是他们的关系式了。而这一步，也是最难的一步，后面我会讲几种类型的题来说。

**第三步骤**：找出**初始值**。学过**数学归纳法**的都知道，虽然我们知道了数组元素之间的关系式，例如 dp[n] = dp[n-1] + dp[n-2]，我们可以通过 dp[n-1] 和 dp[n-2] 来计算 dp[n]，但是，我们得知道初始值啊，例如一直推下去的话，会由 dp[3] = dp[2] + dp[1]。而 dp[2] 和 dp[1] 是不能再分解的了，所以我们必须要能够直接获得 dp[2] 和 dp[1] 的值，而这，就是**所谓的初始值**。

### 1.1 背包 DP

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
cout<< V - res <<endl;

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
            dp[j] = max(dp[j], dp[j - k * a[i]] + k * b[i]);
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

**未AC**

我想到的是反悔贪心（边走边拿，能拿下就把拿到的记录到某个数据结构 A 中；拿不下就反悔，从 A 中用 01背包选反悔的）

实际上就是给状态转移加个判断，满足就可以转移，反之则不能。

对于该题，还要注意：考虑到价值和体积的大小，**背包的第二参量应为价值**，即 $dp_{ij}$ 表示"当遍历到第 i 个物品、得到 j 价值时，花费体积的最小值"

故有：
$$
if(dp[i - 1][j] + w[i] <= (i - 1) * x )\\
dp[i][j] = min(dp[i - 1][j], dp[i - 1][j + v[i]] + w[i])\\
otherwise\\
dp[i][j] = dp[i - 1][j]\\
$$

#### 改变参量的背包

### 1.2 区间 DP

#### LIS（最长单调子序列） $O(n^2)$ 解法

给出模板：
$$
dp[i] = choose_{1 \leq j < i}(change(\ dp[j]\ )),if(pair(i,j)\ can\ move)
$$
[B3637 最长上升子序列 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/B3637)
$$
dp[i] = max_{1 \leq j < i}(dp[j] + 1),if(a[i] > a[j])
$$
[Problem - 1260 (hdu.edu.cn)](https://acm.hdu.edu.cn/showproblem.php?pid=1260)

**未AC**
$$
同下，用dp[i]表示最后一个接的馅饼为i时，总接到的馅饼的最大值\\
dp[i] = max_{1 \leq j < i}(dp[j] + 1),if(diatance(i, j) < \Delta time)\\
这个思路没啥大问题，但是题目要求刚开始时在5的位置...\\
暂时实现不了，换一种dp：\\
dp[i][j]：第i秒位于j时，接到馅饼数的最大值\\
dp[i][j] = max(dp[i - 1][j], dp[i - 1][j + 1], dp[i - 1][j - 1]) + cake[i][j]\\
考虑到边界问题，把dp[i][0]和dp[i][11]设置为-1\\
可用滚动数组优化，但是由于每次小循环时会读取dp[i - 1][j + 1](假定 j 是从1到10遍历)\\
那么如果直接去掉第一维，那么在第一维为i时更新dp[j + 1]之前，已经把dp[j]更新为dp[i][j]了\\
$$

```

```

[P2285 [HNOI2004\] 打鼹鼠 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P2285)
$$
起初我想用x,y,t三个参量，但是时间复杂度O(n^2T)\\
显然超时\\
实际上给出的地鼠序列是按时间排好序的\\
是“可选性最长单调子序列”\\
记dp[i]为最后打第i个地鼠时，打地鼠数的最大值\\
dp[i] = max_{1 \leq j < i}(dp[j] + 1),if(MHTdistance(a[i], a[j]) \leq \Delta time(i, j) )
$$

**曼哈顿距离**：简单说就是走格子数，区别于欧氏距离

```c++
for(int i = 1; i <= m; ++ i)
	for(int j = 1; j < i; ++ j)
		if(MHTdistance(a[i], a[j]) <= abs(a[i].t - a[j].t))
			dp[i] = max(dp[i], dp[j] + 1);
```

[P4310 绝世好题 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P4310)

**未AC**
$$
第一眼感觉是很标准的LIS\\
dp[i] = max_{1 \leq j < i}(dp[j]),if(a[i] \& a[j] \neq 0)\\
emmm光速交了一发，发现有一个点TLE了\\
n 为 1e5...\\
$$



[P1874 快速求和 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1874)
$$
先预处理求出从 i 到 j 的字符串的值 sum[i][j]\\
下面才是 dp：\\
dp[i][k] 表示最后一个为 i、和为 k 时的最少加号数\\
dp[i][k] = min_{1 \leq j \leq i}(dp[j][k - sum[j][i]] + 1)
$$

[Problem - 1160 (hdu.edu.cn)](https://acm.hdu.edu.cn/showproblem.php?pid=1160)

**未AC**
$$
给出数列 v，w，求最长的序列，使得 v [ i ]  > v [ i - 1]，w [ i ] < w [ i - 1],另外要输出一个具体方案
$$

```

```

[Problem - 1260 (hdu.edu.cn)](https://acm.hdu.edu.cn/showproblem.php?pid=1260)
$$
给出数列a，b，分别表示i的权重，i和i+1两个人一起算的权重，求出所有人权重的最小值\\
记dp[i]为i结尾时，权重的集合，取其最小值\\
dp[i] = min(dp[i - 1] + a[i], dp[i - 2] + b[i - 1])\\
$$
[Problem - 1087 (hdu.edu.cn)](https://acm.hdu.edu.cn/showproblem.php?pid=1087)
$$
给出数列a，求和最大的递增子序列\\
记dp[i]为最后一个选i时的最大和\\
dp[i] = max_{1 \leq j < i}(dp[j] + a[i]),if(a[i]>a[j])
$$
[Problem - 1069 (hdu.edu.cn)](https://acm.hdu.edu.cn/showproblem.php?pid=1069)

**未AC**
$$
先按长排好序，再dp宽\\
dp[i] = max_{1 \leq j < i}(dp[j] + h[i]),if(b[i] > b[j])\\
——然后连案例都过不去（\\
题目中三维是可以转动而更改的，并且每种积木都是无限量供应\\
如果我们把每个积木的选择情况全排列出来，每个积木有3！=6种，而且还是无限供应\\
看似很麻烦，但实际上高相同的积木最多只用一次，所以只要把高不同的情况列出来就可以，每个都有3种。\\
然后对长进行降序排序，对宽进行dp，方程不变，但是转移条件改为：\\
a[i] > a[j] \& b[i] > b[j](有a是因为sort不去重\\
另外i的范围扩大到（1，3*n）\\
案例过去了，但是 wa 了
$$


#### 最长公共子序列$O(n^2)$解法


$$
记dp[i][j]为:\\
s1最后一位为s1[i]，s2最后一位为s2[j]时，公共子序列长度最大值\\
dp[i][j] = 
\left\{ 
\begin{array}{}
dp[i - 1][j - 1] + 1, if(s1[i] = s2[j])\\
max(dp[i - 1][j], dp[i][j - 1]),otherwise\\
\end{array}
\right.
$$

```c++
for(int i = 1; i <= size1; ++ i)
	for(int j = 1; j <= size2; ++ j)
		if(s1[i - 1] == s2[j - 1])
			dp[i][j] = dp[i - 1][j - 1] + 1;
		else
			dp[i][j] = max(dp[i][j], max(dp[i - 1][j]， dp[i][j - 1]));
```

#### 最长公共子串$O(n^2)$解法

$$
记dp[i][j]为：\\
以s1[i]，s2[j]结尾时，公共子串最大长度\\
dp[i][j] = \begin{cases}
dp[i - 1][j - 1] + 1, if(s1[i] = s2[j])\\
0, otherwise
\end{cases}
$$

#### 最长的 a 子序列且是 b 子串

想到这样一个问题：
$$
已知字符串a，b，求len(s)_{max}\\
s既是a的子序列，又是b的子串\\
$$

$$
记dp[i][j]为:\\
以a最后一位是a[i]，b最后一位为b[j]时，公共子序列长度最大值\\
dp[i][j] = 
\left\{ 
\begin{array}{}
dp[i - 1][j - 1] + 1, if(a[i] = b[j])\\
dp[i - 1][j],otherwise\\
\end{array}
\right.
$$

```c++
for(int i = 1; i <= n; ++ i)
	for(int j = 1;  j <= m; ++ j)
        if(a[i - 1] == b[j - 1])
            dp[i][j] = dp[i - 1][j - 1] + 1;
		else
            dp[i][j] = dp[i - 1][j];
```



### 1.3 路径 DP

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

### 1.4 暂未分类

#### 最大的连续子段和

[P1115 最大子段和 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1115)

自己做的时候只想出前缀和+穷举，复杂度$O ( n ^ 2 )$.

不要思维定式地用前缀和，nor 潜意识地认为子段和就比单项大。DP的关键在于递推式，所以重要的是**找到局部的递推过程**！

```c++
for(int i = 1; i <= n; ++i)
    dp[i] = max(dp[i - 1] + s[i], s[i]);
```



## 2. 贪心

#### LIS（最长单调子序列）$O(nlogn)$解法

$$
令f[i]为长度为i的最长递增子序列的末尾元素，有：\\
f[i]单调不减（反证可得），故可以二分维护\\
遍历整个原序列，每次都
$$

```

```



#### LIS 的应用

先给出一个结论：**去掉有限个元素的序列，其 LIS 一定不长于原本的 LIS**，反证即可证明

[P1020 [NOIP1999 提高组\] 导弹拦截 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1020)
$$
有两问，第一问先 dp 求出最长子序列，纯粹的LIS\\
第二问就是贪心，找出最少的单调子列数，使之能覆盖原列的每一个元素\\
解法是找出最长的单调递增序列
$$



#### 最大的拼接字符串

[P1012 [NOIP1998 提高组\] 拼数 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1012)

给出数字$a_1, a_2, a_3...$，求出拼接后最大的字符串

下面以两个数字为例：

不是比较$a_1, a_2$的大小

（反例：321 大于 32，但是 32321 的大于  32132）

应该比较$a_1+a_2,a_2+a_1$的大小（这里的 + 表示字符串拼接）

```c++
string a[MAXN], tmp[MAXN];//相同长度时( len(s1+s2) = len(s2+s1) )，数字大小就是字典序大小，所以用字符串来存储和操作

void merge_sort(string a[], int l, int r)
{
    if(l >= r)return ;

    int mid = l + r >> 1;
    merge_sort(a, l, mid);
    merge_sort(a, mid + 1, r);

    int k = 0, i = l, j = mid + 1;
    while(i <= mid && j <= r)
    { 
        if(a[j] + a[i] < a[i] + a[j]) tmp[k ++] = a[i ++];//这里
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

## 3. 字符串

### Part 3.1 哈希

模板

```c++
int hash[MAXN];//hash[i]: 从 s[0] 到 s[i] 的哈希值

int get_hash(const string &s, int l = 0, int r = -1)
{    
    const int p = 126;//进制
    
    if (r == -1)
        r = s.size() - 1;//相当于缺省，未输入l和r时，返回整段的哈希
    if(l)
        return hash[r] - hash[l - 1] * power(p, r - l - 1);//起点不是s[0]的子段
    //求s[0~r]的哈希
    hash[r] = 0;
    for (int i = l; i <= r; ++i)
        hash[r] = hash[r] * p + s[i];
    return hash[r];
}
```

[P2580 于是他错误的点名开始了 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P2580)
$$
别人说是字典树问题，我只会无脑哈希（\\
n = 1e4，m = 1e5，S = 50\\
O(nS)算hash值，O(mSlogn)查找\\
有空拿字典树做做
$$

```c++
for(int i = 1; i <= m; ++ i)
{
    cin>>s;
    int hash = get_hash( s, 0, s.size() - 1);
    if(st.find( hash ) == st.end())		{}
    else	{}
}
```

[P3879 [TJOI2010\] 阅读理解 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P3879)



## 4. 数学

**关注点通常是 "如何实现 / 加速计算" 而非 "如何手算"**

### 4.1 基础算法

#### GCD/LCM（最大公因数/最小公倍数）

求 a, b 的最大公因数/最小公倍数

```c++
int gcd (int a, int b)
{
    if (!b) return a;
    return gcd (b, a % b);
}
```

```c++
long long lcm = (a * b) / gcd(a,b);
```

#### fp （快速幂）

求 a ^ b mod p

[P1226 【模板】快速幂 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P1226)

```c++
long long fp (long long a, long long b, long long p) {
    long long res = 1;
    a % = p;//优化，减小不必要的运算
    while (b) {
        if (b & 1) res = res * a % p;//b & 1 <==> b != 1
        b >>= 1;
        a = (a * a) % p;
    }
    return res;
}
```

#### EXGCD（拓展欧几里得算法）

求解二元一次不定方程 ax + by = (a, b)

```c++
int exgcd(int a, int b ,int &x ,int &y)
{
    if(!b)
    {
        x = 1, y = 0;
        return a;
    }
    int d = exgcd(b, a%b, y, x);
    y -= (a / b * x);
    return d;
}
```





### 4.2 数论 & 组合数学

##### 质数判断

判断单个素数 $O(\sqrt{n})$

```c++
bool isPrime(int a) {
    if (a <= 1)	return false;
    if (a == 2 || a == 3)	return true;
    if (a % 2 == 0 || a % 3 == 0)	return false;
    
    int q = sqrt(a);
    for (int i = 5; i <= q; i += 6) 
        if (a % i == 0 || a % (i + 2) == 0) 
            return false;
    return true;
}
```



**欧拉筛**

$o(n)$

```c++
vector<int> prime;
bool not_prime[MAXN + 5];

for(int i = 2; i <= MAXN; ++ i)
{
    if(! not_prime[i])  prime.push_back(i);
    for(int j = 0; j < prime.size() && prime[j] * i < MAXN; ++ j)
    {
        not_prime[i * prime[j]] = true;
        if(i % prime[j] == 0)   break;   
    }    
}
```

### 4.3 博弈论

## 5. 数据结构

| 数据结构 |                             功能                             |
| :------: | :----------------------------------------------------------: |
|  ST 表   |                      $O(1)$查询区间最值                      |
|   莫队   | $O((n + q) \sqrt{n})$完成所有离线静态的区间查询<br>处理复杂的区间查询时容易实现 |
|  并查集  |                        $O(1)$合并查询                        |
| 树状数组 |                        mini 版线段树                         |
|  线段树  |                      $O(logn)$增删查改                       |
|          |                                                              |
|          |                                                              |
|          |                                                              |
|          |                                                              |



### 5.1 静态数据结构

前缀和数组见"基础算法"

#### ST 表



#### 莫队



### 5.2 栈 & 队列

#### 单调栈

单调栈能解决的问题就很少，基本上都是求一个数列中符合要求的最近的数，给出模板：

```c++
for(int i = 1; i <= n; ++ i)//顺序要因题而异
{
    //...
    while(!st.empty() && a[i] is priority than a[st.top()] )
        st.pop();
    //...
    st.push(i);
}
```

[P5788 【模板】单调栈 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P5788)

```c++
for(int i = 1; i <= n; ++ i)//顺序要因题而异
{
    while(!st.empty() && a[i] is priority than a[st.top()] )
        st.pop();
    if(st.empty())//把栈里所有值都弹出了，说明这个值是最优先的
        ans[i] = 0;
    else
        ans[i] = st.top().order;
    st.push(i);
}
```

[D - Buildings (atcoder.jp)](https://atcoder.jp/contests/abc372/tasks/abc372_d#:~:text=D - Buildi)

```c++
for(int i = n; i >= 1; -- i)
{
    ans[i] = st.size();
    while(!st.empty() && h[i] > h[st.top()])
    {
        st.pop();
    }
    st.push(i);
}
```



#### 单调队列

值得注意地是，单调队列是一个双向队列。单调队列和滑动窗口紧密相连，也能用来优化某些 dp，给出模板：

```c++
for(int i = 1; i <= n; ++ i)
{
	if(!q.empty() && q.front() should be over)
		q.pop_front();
    //设窗口大小为 k，显然当 i <= k 时要一直塞入，直到窗口已经有 k 个值时才开始运作
	while(!q.empty() && i > k && q.back().data is priority than a[i].data)
		q.pop_back();
	q.push_back(a[i]);
	ans[i] = q.front().data;
}
```



### 5.3 并查集

用树的结构来合并两个集合，也能以近乎$O(1)$的速度查询某个值是否在该集合中，给出模板：

```c++
int p[MAXN];//下标为对应数据，值为对应数据的父节点；只有一个数据等于自己的父节点时，该数据才为祖宗节点

int find(int x)
{
    if(p[x] != x) p[x] = find(p[x]);
    return p[x];
}

void merge(int a, int b)
{
	p[ find(a) ] = find( p[b] );
}
```



**并查集的访问**

同一个集合如何只访问一次

```c++
merge(u, v);
for(int i = 1; i <= n; ++ i)
{
    int fx = find(i);
    if( done[fx] )	continue;
    done[fx] = true;
    //...
}
```



### 5.4 树状数组

**线段树的泛用性大于树状数组，但是树状数组比线段树简单**

对于一个数组，差分可以$O(1)$修改区间（加同一个数）；前缀和可以$O(1)$查询区间。（对单点的修改和查询，可视为弱化的区间修改和查询）

但是如果两者都需要时，就要用到能在$O(logn)$时间内完成查询、$O(1)$时间完成加区间加数的树状数组

对于区间查询和单点修改

```c++
int lowbit(int x)//c[x]的前缀和范围
{
	return x & -x;
}

int getnum(int x)//从a[1]到a[x]的和查询
{
    int ans = 0;
    while(x > 0)
    {
		ans += c[x];
        x -= lowbit(x);
    }
   	return ans;
}

void add(int x, int k)
{
    while(x <= n)
    {
        c[x] += k;
        x += lowbit(x);
    }
}

//建树
for(int i = 1; i <= a.size(); ++ i)
    add(i, a[i]);
```



#### 基于树状数组的 LIS



### 5.5 线段树

线段树可以在$O(logn)$的时间复杂度内实现单点修改、区间修改、区间查询（区间求和，求区间最大值，求区间最小值)等操作。

[P3372 【模板】线段树 1 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P3372)

```c++
//流程：建树从
int n;
int tree[N * 4], a[N], tag[N * 4];

//求子节点
int ls(int p)	{return p << 1;}
int rs(int p)	{return p << 1 | 1;}

void addtag(int p, int pl, int pr, int d)
{
    tag[p] += d;//给结点 p 打 tag
    tree[p] += d * (pr - pl + 1);//更新 tree
}

//更新父节点
void push_up(int p)
{
    tree[p] = tree[ ls(p) ] + tree[ rs(p) ];
    //tree[p] = change( tree[ ls(p) ], tree[ rs(p) ] );
    //change 为 +、min、max等满足交换律的运算
}

//更新子节点（的tag）
void push_down(int p, int pl, int pr)
{
    if(tag[p])
    {
        int pm = (pl + pr) >> 1;
        addtag(ls(p), pl, pm, tag[p]);
        addtag(rs(p), pm + 1, pr, tag[p]);
        tag[p] = 0;//自己的 tag 归零
    }
}

//建树
void build(int p = 1, int pl = 1, int pr = n)//用法：build();
{
    // 对 [pl, pr] 区间建立线段树,当前根的编号为 p
    if (pl == pr) {
        tree[p] = a[pl];
        return;
    }
    int pm = (pl + pr) >> 1;
    build(ls(p), pl, pm);
    build(rs(p), pm + 1, pr );

    push_up(p);
}

// [l, r] 为查询区间, [pl, pr] 为当前节点包含的区间, p 为当前节点的编号	
int query(int l, int r, int p = 1, int pl = 1, int pr = n)//调用方法:query(l, r)
{
    // 当前区间为询问区间的子集时直接返回当前区间的和
    if (l <= pl && pr <= r)
        return tree[p];
    push_down(p, pl, pr);
    int pm = (pl + pr) >> 1;
    int ans = 0;

    // 如果左/右儿子代表的区间与询问区间有交集, 则递归查询左/右儿子
    if (l <= pm)
        ans += query(l, r, ls(p), pl, pm);
    if (r > pm)
        ans += query(l, r, rs(p), pm + 1, pr);
    //同上，change
    return ans;
}

//维护线段树
//区间 [l, r] 中每个元素加 d，后三个参数是递归时用的
void update(int l, int r, int d, int p = 1, int pl = 1, int pr = n)
{
    if (l <= pl && pr <= r) {
        addtag(p, pl, pr, d);
        return;
    }
    push_down(p, pl, pr);//把tag传给子树
    int pm = (pl + pr) >> 1;
    if(l <= pm)
        update(l, r, d, ls(p), pl, pm);
    if(r > pm)
        update(l, r, d, rs(p), pm + 1, pr);
    push_up(p);
}
```

## 6. 计算几何

### 6.1 二维几何

由于精度问题，需要一个相等判断函数

```c++
const double pi = acos(-1.0);
const double eps = 1e-8;

int sign(double x)
{
	if( fabs(x) < eps )	return 0;
    else	return x < 0 ? -1 : 1;
}

//比较两个浮点数
int dcmp(double x, double y)
{
    if( fabs(x - y) < eps)	return 0;
    else	return x < y ? -1 : 1;
}
```

#### 6.1.1 点 & 向量

```c++
struct point{
    double x, y;
    point();
    point(double x, double y):x(x), y(y){}
};

//两点间距离(方便起见用了 hypot，也可以用 sqrt)
double getDistance(point A, point B)
{	return hypot(A.x - B.x, A.y - B.y);}

//向量用从原点指向的点表示
typedef point Vector;
//重载值、点和向量之间的运算符
point operator + (point B)
{	return point(x + B.x, y + B.y);}
point operator - (point B)
{	return point(x - B.x, y - B.y);}
point operator * (double k)
{	return point(x * k, y * k);}
point operator / (double k)
{	return point(x / k, y / k);}
bool operator == (point B)
{	return sgn(x - B.x) == 0 && sgn(y - B.y) == 0;}
```

#### 6.1.2 点积和叉积

```c++
double len(Vector A)
{	return sqrt( A.x * A.x + A.y * A.y);}
double len2(Vector A)
{	return A.x * A.x + A.y * A.y;}
//点乘
double dot(Vertor A, Vertor B)
{	return A.x * B.x + A.y * B.y;}

//夹角
double angle(Vector A, Vector B)
{	return acos( dot(A, B) / len(A) / len(B));}

//叉乘
double cross(Vertor A, Vertor B)
{	return A.x * B.y - A.y * B.x;}
//若A × B > 0，则 B 在 A 的逆时针方向，反之为顺时针；等于 0 说明共线

//已知三点 A B C，求构成平行四边形的面积
double area2(point A, point B, point C)
{	return cross(B - A, C - A);}

//向量 A 逆时针旋转角度 rad
Vector rotate(Vector A, double rad)
{
    double c = cos(rad), s = sin(rad);
    return Vector(A.x * c - A.y * s, A.x * s + A.y * c);
}

//单位法向量
Vector normal(Vector A)
{	return Vector( -A.y, -A.x) / len(A);}

//是否共线
bool parallel(Vector A, Vector B)
{	return sgn(cross(A, B) == 0);}
```

#### 6.1.3 点和线

```c++
struct line{
    point p1, p2;
    line(){};
    line(point p1, point p2):p1(p1), p2(p2){};
    line(point p, double angle)//点斜式
    {
        p1 = p;
        if(sgn(angle - pi / 2) == 0)
            p2 = p1 + point(0, 1);
        else
            p2 = p1 + point(1, tan(angle));
    }
    line(double a, double b, double c)//一般式
    {
        if(sgn(a) == 0)
        {
            p1 = point(0, -c / b);
            p2 = point(1, -c / b);
        }
        else if(sgn(b) == 0)
        {
            p1 = point(0, -c / b);
            p2 = point(1, -c / b);
        }
        else
        {
            p1 = point(0, -c / b);
            p2 = point(1, -(c + a) / b);
        }
    }
};

//可以用两个点表示线段
typedef line segment;

//点和直线关系
int point_line_relation(point p, line v)
{
    int c = sgn( cross(p - v.p1, v.p2 - v.p1));
    if(c < 0)	return 1;
    if(c > 0)	return 2;
    return 0;
}

//点和线段关系（是否在线段上）
bool point_on_seg(point p, line v)
{	
    bool m = (cross(p - v.p1, v.p2 - v.p1) == 0);//PA 和 PB 是否共线
    bool n = (sgn(dot(p - v.p1, p - v.p2)) <= 0);//PA 和 PB 是否反向（点积小于 0）
    return m && n;
}
```



## 7. 图论

### 7.1 建图和遍历

n 个点，m 条边，点 u 的入度 $d^-(u)$，出度$d^+(u)$。

#### 建图

**邻接矩阵**

一般只用于稠密图

使用一个二维数组 adj 来存边，其中 adj [u] [v]  表示 u  到 v 的边权，为 0 表示不存在。只适用于没有重边（或重边可以忽略）的情况。

优点：查询一条边是否存在$O(1)$。

缺点：空间复杂度$O(n^2)$

**邻接表**

使用最多

使用一个支持动态增加元素的数据结构构成的数组，如 

`vector<int> adj[MAXN];`

`adj [u]`存的是点 u 的所有出边信息（终点、边权等）

```c++
// 对于每个点k，开一个单链表，存储k所有可以走到的点。
//h[k]存储这个单链表的头结点; e[k]存储结点的值; ne[k]存储结点的 next 
int h[N], e[N], ne[N], idx;

// 添加一条边a->b
void add(int a, int b)
{
    e[idx] = b, ne[idx] = h[a], h[a] = idx ++ ;
}

// 初始化
idx = 0;
memset(h, -1, sizeof h);
```



```c++
struct edge{
    int from, to, w;
    edge(int a, int b, int c){
        from = a;
        to = b;
        w = c;
    }
};

vector<edge> e[maxn];

//...

    while(m --)
    {
        int u, v, w;
        cin>>u>>v>>w;
        e[u].push_back(edge(u, v, w));
        //双向图加下面代码
        //e[v].push_back(edge(v, u, w));
    }
```



查询是否存在从 u 到 v 的边：$O(d^+(u))$；若有多次查询，可以先排序再二分查找，$O(log(d^+(u)))$

遍历整张图：$O(n+m)$

**链式前向星**

本质上是用链表实现的邻接表

```c++
// head[u] 和 cnt 的初始值都为 -1
void add(int u, int v) {
  nxt[++cnt] = head[u];  // 当前边的后继
  head[u] = cnt;         // 起点 u 的第一条边
  to[cnt] = v;           // 当前边的终点
}

// 遍历 u 的出边
for (int i = head[u]; i != -1; i = next[i]) {
  int v = to[i];
}

int n, m;
vector<bool> vis;
vector<int> head, nxt, to;

void add(int u, int v) {
  nxt.push_back(head[u]);
  head[u] = to.size();
  to.push_back(v);
}

bool find_edge(int u, int v) {
  for (int i = head[u]; ~i; i = nxt[i]) {  // ~i 表示 i != -1
    if (to[i] == v) {
      return true;
    }
  }
  return false;
}

void dfs(int u) {
  if (vis[u]) return;
  vis[u] = true;
  for (int i = head[u]; ~i; i = nxt[i]) dfs(to[i]);
}
```

### 7.2 最短路



|       应用       |  边权条件  |      算法      |    时间复杂度     |
| :--------------: | :--------: | :------------: | :---------------: |
|  **单点到单点**  |   无边权   |     A*算法     | $<O((m + n)logn)$ |
|                  |   无边权   |    双向广搜    | $<O((m + n)logn)$ |
|                  |   无边权   |  贪心最优搜索  |    $<O(m + n)$    |
| **单点到所有点** |   无边权   |      BFS       |    $O(m + n)$     |
|                  |   非负数   |    Dijkstra    | $O((m + n)logn)$  |
|                  | 允许有负数 |      SPFA      |     $<O(mn)$      |
|  **所有点之间**  | 允许有负数 | Floyd-Warshall |    $O(n ^ 3)$     |



#### dijkstra 算法

稀疏图$O(nlogn)$, 稠密图$O(n^2)$

```c++
//注释:输出最短路径
struct edge{
    int from, to, w;
    edge(int a, int b, int c){
        from = a;
        to = b;
        w = c;
    }
};

vector<edge> e[maxn];

struct node{
    int id, n_dis;
    node(int b, int c){
        id = b;
        n_dis = c;
    }
    bool operator <(const node & a)const{
        return n_dis > a.n_dis;
    }
};

int n, m;
int dis[maxn];
bool done[maxn];
//int pre[maxn];

//void printPath(int s, int t)
//{
//    if(s == t) { cout<<s<<' '; return; }
//    printPath(s, pre[t]);
//    cout<<t<<' ';
//}

void dijkstra(){
    int s = 1;
    for(int i = 1; i <= n; ++ i) { dis[i] = INF; done[i] = false; }
    dis[s] = 0;
    priority_queue<node> q;
    q.push(node(s, dis[s]));

    while( !q.empty())
    {
        node u = q.top();   q.pop();
        if(done[u.id])  continue;
        done[u.id] = true;
        for(int i = 0; i < e[u.id].size(); ++ i)
        {
            edge y = e[u.id][i];
            if(done[y.to]) continue;
            if(dis[y.to] > y.w + u.n_dis)
            {
                dis[y.to] = y.w + u.n_dis;
                q.push(node(y.to, dis[y.to]));
                //pre[y.to] = u.id;
            }
        }
    }
    //printPath(s, n);
}

void solve(){
    cin>>n>>m;
    for(int i = 1; i <= n; ++ i)
    {
        e[i].clear();
    }

    while(m --)
    {
        int u, v, w;
        cin>>u>>v>>w;
        e[u].push_back(edge(u, v, w));
        //双向图加下面代码
        //e[v].push_back(edge(v, u, w));
    }

    dijkstra();

    for(int i = 1; i <= n; ++ i)
    {
        if(dis[i] >= INF)
            cout<<-1;
        else
            cout<<dis[i]<<' ';
    }
}
```



#### 分层图最短路

**难在建图**

**加速点的最短路**

起始节点的实图和虚图相连, 加速点的实图单向连接虚图中的邻居节点

[Problem - 2014E - Codeforces](https://codeforces.com/problemset/problem/2014/E#:~:text=E. Rendez-vous de Marian et Robin. In)
$$
图中有 h 个节点有马, 骑马后速度加倍, 两个人从 1 和 n 出发, 问相遇的最短时间(只能在节点上相遇)
$$

```c++
//1 MB == 1e6 int
#include <bits/stdc++.h>
#define int long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define CLEAR(a) {memset(a, 0, sizeof(a));}
using namespace std;//上面的都别动
#define endl '\n'//注意该宏也作用于check（无法刷新输出缓存区
const int INF = 0x3f3f3f3f3f3f3f3f;
const int inf = 0x3f3f3f3f;
const int maxn = 2e5 + 10;

struct edge{
    int from, to, w;
    edge(int a, int b, int c){
        from = a;
        to = b;
        w = c;
    }
};

vector<edge> e[maxn * 2];

struct node{
    int id, n_dis;
    node(int b, int c){
        id = b;
        n_dis = c;
    }
    bool operator <(const node & a)const{
        return n_dis > a.n_dis;
    }
};

int n, m, h;
int dis[maxn * 2], dis2[maxn * 2];
bool done[maxn * 2], done2[maxn * 2];
bool haveHorse[maxn];

void dijkstra(){
    int s = 1;
    for(int i = 1; i <= n * 2; ++ i) { dis[i] = INF; done[i] = false; }
    dis[s] = 0;
    priority_queue<node> q;
    q.push(node(s, dis[s]));

    while( !q.empty())
    {
        node u = q.top();   q.pop();
        if(done[u.id])  continue;
        done[u.id] = true;
        for(int i = 0; i < e[u.id].size(); ++ i)
        {
            edge y = e[u.id][i];
            if(done[y.to]) continue;
            if(dis[y.to] > y.w + u.n_dis)
            {
                dis[y.to] = y.w + u.n_dis;
                q.push(node(y.to, dis[y.to]));
                //pre[y.to] = u.id;
            }
        }
    }
    //printPath(s, n);
}

void dijkstra2(){
    int s = n;
    for(int i = 1; i <= n * 2; ++ i) { dis2[i] = INF; done2[i] = false; }
    dis2[s] = 0;
    priority_queue<node> q;
    q.push(node(s, dis2[s]));

    while( !q.empty())
    {
        node u = q.top();   q.pop();
        if(done2[u.id])  continue;
        done2[u.id] = true;
        for(int i = 0; i < e[u.id].size(); ++ i)
        {
            edge y = e[u.id][i];
            if(done2[y.to]) continue;
            if(dis2[y.to] > y.w + u.n_dis)
            {
                dis2[y.to] = y.w + u.n_dis;
                q.push(node(y.to, dis2[y.to]));
                //pre[y.to] = u.id;
            }
        }
    }
    //printPath(s, n);
}

void solve(){
    CLEAR(haveHorse);
    cin>>n>>m>>h;
    for(int i = 1; i <= n * 2; ++ i)
    {
        e[i].clear();
    }

    for(int i = 1; i <= h; ++ i)
    {
        int buf;
        cin>>buf;
        haveHorse[buf] = true;
    }

    while(m --)
    {
        int u, v, w;
        cin>>u>>v>>w;
        e[u].push_back(edge(u, v, w));
        e[v].push_back(edge(v, u, w));
        e[u + n].push_back(edge(u + n, v + n, w / 2));
        e[v + n].push_back(edge(v + n, u + n, w / 2));
        if(haveHorse[u]) e[u].push_back(edge(u, v + n, w / 2));
        if(haveHorse[v]) e[v].push_back(edge(v, u + n, w / 2));
    }

    dijkstra();
    dijkstra2();

    dis[1 + n] = 0;
    dis2[n + n] = 0;
    int ans = INF;
    cout<<endl;
    for(int i = 1; i <= n; ++ i)
    {
        int min1 = min(dis[i], dis[n + i]);
        int minn = min(dis2[i], dis2[n + i]);
        int now = max(min1, minn);
        ans = min(ans, now);
    }
    if(ans >= INF)
        cout<<-1<<endl;
    else
        cout<<ans<<endl;
}

signed main(){
    ISO;
    int t = 1;
    cin>>t;
    while(t--)
        solve();
    return 0;
}

```

**多层不同权值图**

[小雨坐地铁 (nowcoder.com)](https://ac.nowcoder.com/acm/problem/26257)
$$
建 m + 1层图, 其中第 1 层作为中转站, 这样能避免 (m + 1)! 的接图操作, 只要每个图与第一层相连即可\\
$$


```c++
//1 MB == 1e6 int
#include <bits/stdc++.h>
#define int long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define CLEAR(a) {memset(a, 0, sizeof(a));}
using namespace std;//上面的都别动
//#define endl '\n'//注意该宏也作用于check（无法刷新输出缓存区
const int INF = 0x3f3f3f3f3f3f3f3f;
const int inf = 0x3f3f3f3f;
const int N = 1e3 + 10;
const int M = 5e2 + 10;
const int maxn = M * N;

struct edge{
    int from, to, w;
    edge(int a, int b, int c){
        from = a;
        to = b;
        w = c;
    }
};

vector<edge> e[maxn];

struct node{
    int id, n_dis;
    node(int b, int c){
        id = b;
        n_dis = c;
    }
    bool operator <(const node & a)const{
        return n_dis > a.n_dis;
    }
};

int n, m, begin1, end1;
int a[M], b[M], c[M];
int dis[maxn];
bool done[maxn];

void dijkstra(){
    int s = begin1;
    for(int i = 1; i <= n * (m + 1); ++ i) { dis[i] = INF; done[i] = false; }
    dis[s] = 0;
    priority_queue<node> q;
    q.push(node(s, dis[s]));

    while( !q.empty())
    {
        node u = q.top();   q.pop();
        if(done[u.id])  continue;
        done[u.id] = true;
        for(int i = 0; i < e[u.id].size(); ++ i)
        {
            edge y = e[u.id][i];
            if(done[y.to]) continue;
            if(dis[y.to] > y.w + u.n_dis)
            {
                dis[y.to] = y.w + u.n_dis;
                q.push(node(y.to, dis[y.to]));
            }
        }
    }
}

void solve(){
    cin>>n>>m>>begin1>>end1;
    for(int i = 1; i <= n * (m + 1); ++ i)
    {
        e[i].clear();
    }

    for(int i = 1; i <= m; ++ i)//第i + 1层图
    {
        cin>>a[i]>>b[i]>>c[i];
        int last, now;
        cin>>now;
        for(int j = 1; j < c[i]; ++ j)
        {
            last = now;
            cin>>now;
            int last1 = last + i * n;
            int now1 = now + i * n;
            e[last1].push_back(edge(last1, now1, b[i]));
            e[now1].push_back(edge(now1, last1, b[i]));
        }
    }

    for(int i = 1; i <= n; ++ i)
    {
        for(int j = 1; j <= m; ++ j)
        {
            int p = i + j * n;
            e[i].push_back(edge(i, p, a[j]));
            e[p].push_back(edge(p, i, 0));
        }
    }

    dijkstra();

    int ans = dis[end1];
    if(ans >= INF)
        cout<<-1<<endl;
    else
        cout<<ans<<endl;
}

signed main(){
    ISO;
    int t = 1;
//    cin>>t;
    while(t--)
        solve();
    return 0;
}

```



[P4568 [JLOI2011\] 飞行路线 - 洛谷 | 计算机科学教育新生态 (luogu.com.cn)](https://www.luogu.com.cn/problem/P4568)
$$
不同层之间单向链接, 否则上下横跳可以0权到达所有点
$$


```c++
//1 MB == 1e6 int
#include <bits/stdc++.h>
#define int long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define CLEAR(a) {memset(a, 0, sizeof(a));}
using namespace std;//上面的都别动
#define undirectedEdge(u, v, w) {e[u].push_back(edge(u, v, w)); e[v].push_back(edge(v, u, w));}
//#define endl '\n'//注意该宏也作用于check（无法刷新输出缓存区
const int INF = 0x3f3f3f3f3f3f3f3f;
const int inf = 0x3f3f3f3f;
const int maxn = 2e5 + 10;

struct edge{
    int from, to, w;
    edge(int a, int b, int c){
        from = a;
        to = b;
        w = c;
    }
};

vector<edge> e[maxn];

struct node{
    int id, n_dis;
    node(int b, int c){
        id = b;
        n_dis = c;
    }
    bool operator <(const node & a)const{
        return n_dis > a.n_dis;
    }
};

int n, m, k;
int s, t;
int dis[maxn];
bool done[maxn];
//int pre[maxn];

//void printPath(int s, int t)
//{
//    if(s == t) { cout<<s<<' '; return; }
//    printPath(s, pre[t]);
//    cout<<t<<' ';
//}

void dijkstra(){
//    int s = 1;
    for(int i = 0; i <= n * (k + 1); ++ i) { dis[i] = INF; done[i] = false; }
    dis[s] = 0;
    priority_queue<node> q;
    q.push(node(s, dis[s]));

    while( !q.empty())
    {
        node u = q.top();   q.pop();
        if(done[u.id])  continue;
        done[u.id] = true;
        for(int i = 0; i < e[u.id].size(); ++ i)
        {
            edge y = e[u.id][i];
            if(done[y.to]) continue;
            if(dis[y.to] > y.w + u.n_dis)
            {
                dis[y.to] = y.w + u.n_dis;
                q.push(node(y.to, dis[y.to]));
                //pre[y.to] = u.id;
            }
        }
    }
    //printPath(s, n);
}

void solve(){
    cin>>n>>m>>k;
    cin>>s>>t;
    for(int i = 0; i <= n; ++ i)
    {
        e[i].clear();
    }

    while(m --)
    {
        int u, v, w;
        cin>>u>>v>>w;
        undirectedEdge(u, v, w)
        for(int i = 1; i <= k; ++ i)
        {
            int nowv = v + i * n, nowu = u + i * n;
            int lastv = nowv - n, lastu = nowu - n;
            undirectedEdge(nowu, nowv, w);

            e[lastu].push_back(edge(lastu, nowv, 0));
            e[lastv].push_back(edge(lastv, nowu, 0));
        }
    }

    dijkstra();

    int ans = INF;
    
    for(int i = 0; i <= k; ++ i)
    {
        ans = min(ans, dis[t + i * n]);
    }

    cout<<ans<<endl;
}

signed main(){
    ISO;
    int t = 1;
//    cin>>t;
    while(t--)
        solve();
    return 0;
}

```



### 7.3 拓扑排序

给DAG（有向无环图）的节点排序

#### 基于 BFS 的拓扑排序

1. 找到所有入度为 0 的点，放入队列。若找不到入度为 0 的点，说明这个图不是 DAG，不存在拓扑排序
2. 弹出队首 a，a 的所有邻居点入度减 1，入度减为 0 的邻居点 b 入队，没有减为 0 的点不能入队
3. 重复操作 2，直到队列为空

​	**拓扑排序无解的判断**：如果队列已空，但是还有点未入队，那么这些点的入度都不为 0，说明图不是 DAG，不存在拓扑排序。拓扑排序无解，说明图上存在环，利用这一点可以找到图上的环路。用拓扑排序能找到这个环：没有进人队列的点,就是环路上的点。

​	**要求输出字典序最小的排序**：队列改为优先队列即可。



#### 基于 DFS 的拓扑排序

​	从入度为 0 的点开始 DFS，则 DFS 递归的顺序就是一个**逆序**的拓扑排序。为了顺序输出，可以先把 DFS 的输出存储到栈里，然后在从栈里输出。

​	另外，为了处理方便，想象有一个虚拟点，单向连接到了所有其它点。实际编程时，并不需要处理这个虚拟点，只要在主程序中把每个点轮流 DFS 一遍即可——相当于显式地递归了虚拟点的所有的下一层点。

**判断是否为 DAG**：图不是 DAG，说明有环。那么在递归时,会出现回退边。记录每个点的状态，如果 dfs 函数递归到某个点时发现它仍在前面的递归中没有处理完毕，说明存在回退边,不存在拓扑排序。

```c++

```

​	以上两种方法都需要遍历所有的边和点，复杂度均为$O(n+m)$。若要输出所有的排序，使用  DFS；若只输出一个排序（一般是字典序最小的排序），则使用 BFS。

## 8. 杂项

#### 定积分

**自适应辛普森法**

​	辛普森法的思想是将被积区间分为若干小段，每段套用二次函数的积分公式进行计算；为了在保证精度的前提下改进速度，我们每次判断当前段和二次函数的相似程度，如果足够相似的话就直接代入公式计算，否则将当前段分割成左右两段递归求解，这就是自适应。

```c++
double simpson(double l, double r) {
    double mid = (l + r) / 2;
    return (r - l) * (f(l) + 4 * f(mid) + f(r)) / 6;  // 辛普森公式
}

double asr(double l, double r, double eps, double ans, int step) {
    double mid = (l + r) / 2;
    double fl = simpson(l, mid), fr = simpson(mid, r);
    // 足够相似的话就直接返回
    if (abs(fl + fr - ans) <= 15 * eps && step < 0)
        return fl + fr + (fl + fr - ans) / 15;  
    // 否则分割成两段递归求解
    return asr(l, mid, eps / 2, fl, step - 1) + asr(mid, r, eps / 2, fr, step - 1); 
}

double calculus(double l, double r, double eps) {
    return asr(l, r, eps, simpson(l, r), 12);
}
```



#### 启发式合并

[【笔记】启发式合并 - 7KByte - 博客园 (cnblogs.com)](https://www.cnblogs.com/7KByte/p/16143270.html)

给定 n 个集合，每个集合开始只有 1 个数，每次合并两个集合

直接做显然是 $O(n^{2})$ 

每次合并都选择较小的一个集合，将它合并到较大的集合上

看上去本质上没有改变，但时间复杂度降低至 $O(nlogn)$. 因为对于每个元素，每次合并一定是从小的集合合并到大的集合，所以合并后的集合大小一定翻倍，那么一个元素最多移动 logn 次，总的时间复杂度为 $O(nlogn)$

[E - K-th Largest Connected Components (atcoder.jp)](https://atcoder.jp/contests/abc372/tasks/abc372_e?lang=en)
$$
维护连通块首选并查集, 每个点开一个set记录联通点\\
连边时把set合并到父节点中,查询时直接找父节点的set
$$
优化:
$$
set合并需要遍历整个集合,可能会造成重复插入, 最坏情况下时间复杂度为O(n^{2}logn + qlogn)\\
使用按大小合并, 每次合并都把小集合插入到大集合\\
每次合并集合大小至少翻倍, 每个元素最多合并logn次,时间复杂度O(nlogn + qlogn)
$$


```c++
//1 MB == 1e6 int
#include <bits/stdc++.h>
#define int long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define CLEAR(a) {memset(a, 0, sizeof(a));}
using namespace std;//上面的都别动
#define undirectedEdge(u, v, w) {e[u].push_back(edge(u, v, w)); e[v].push_back(edge(v, u, w));}
#define endl '\n'//注意该宏也作用于check（无法刷新输出缓存区
const int INF = 0x3f3f3f3f3f3f3f3f;
const int inf = 0x3f3f3f3f;
const int maxn = 2e5 + 10;

set<int> s[maxn];

int n, q;
int p[maxn];

int find(int x)
{
    if(p[x] != x) p[x] = find(p[x]);
    return p[x];
}

void merge(int a, int b)
{
    p[ find(a) ] = find( p[b] );
}

void solve(){
    cin>>n>>q;
    for(int i = 1; i <= n; ++ i)
    {
        p[i] = i;
        s[i].insert(i);
    }

    while(q --)
    {
        int buf, u, v, k;
        cin>>buf;
        if(buf == 1)
        {
            cin>>u>>v;
            int fu = find(u);
            int fv = find(v);
            if(fu == fv)    continue;
            
            if(s[fu].size() > s[fv].size())
                swap(fv, fu);
            
            for(int tt : s[fu])
                s[fv].insert(tt);
            merge(fu, fv);
        }
        else if( buf == 2)
        {
            cin>>v>>k;
            int fv = find(v);
            if(s[fv].size() < k)
            {
                cout<<-1<<endl;
            }
            else
                cout<<*prev(s[fv].end(), k)<<endl;
        }
    }
}

signed main(){
    ISO;
    int t = 1;
//    cin>>t;
    while(t--)
        solve();
    return 0;
}

```

### 







## PS 训练题

[CUGBACM23级春季学期训练#21A ](https://vjudge.net/contest/627077#problem/C)

$n<500$,  给出数组$x_i<500\ (i=2,3,...n)$,  找出数组$a_i<1e9\ (i=1,2,3,...n)$,  使得$x_i=a_i\ mod\ a_{i-1}$.

[E - Max/Min](https://atcoder.jp/contests/abc356/tasks/abc356_e)

暴力要 $ o(n^2) $ ,考虑优化

* 暴力时每两个元素都要比较一次，不如进行排序预处理，剩下的问题等价于：
  $$
  给定单调递增数列a， 求\Sigma \Sigma \lfloor \frac{a_i}{a_j} \rfloor
  $$
  之后优化如下：

```c++
sort(a + 1, a + n + 1, cmp);

int ans = 0, part = 0;
for(int i = 1; i < n; ++ i)
{
	if(a[i] == a[i - 1])//相同项不再重复计算
	{
        part --;
        ans += part;
        continue;
    }
    
    // last 是前一段的 l
    int l, r;
    int last = i, maxn = a[n] / a[i];
    part = 0;
    
    for(int k = 1; k <= maxn; ++ k)//有点像完全背包的第二层循环？
    {
        l = last;
        r = n;
            
        while(l < r  && r <= n)//r <= n 是防止二分时越界，不过貌似没用？
        {
            int mid = l + r + 1 >> 1;
            if(a[mid] < (k + 1) * a[i])
                l = mid;
            else
                r = mid - 1;
        }
        part += k * (l - last);
        last = l;
    }

    ans += part;
}
```

[Problem - 1989C - Codeforces](https://codeforces.com/problemset/problem/1989/C)
$$
两部电影，每部每人都有态度1,0和-1\\
现在每个人可以对一部电影作出评分，求两电影总评中较小值的最大值\\
$$

$$
分成两种：a_i = b_i 或者 a_i \neq b_i\\
a_i \neq b_i 时，a_i 和 b_i 至少有一个非负，故加最大的即可\\
a_i = b_i 时，a_i 和 b_i 同为-1或1(0的情况没有作用，直接舍去\\
此时小的数加1，大的数加-1即可\\
当然要先处理完第一种情况，才能知道谁大谁小
$$



```c++
int cntp = 0, cntn = 0;
int ans1 = 0, ans2 = 0;
for(int i = 1; i <= n; ++ i)
{
    if(a[i] == b[i])
    {
        if(a[i] == 1)
            ++ cntp;
        else if(a[i] == -1)
            ++ cntn;
    }
    else
    {
        if(a[i] > b[i])
            ans1 += a[i];
        else
            ans2 += b[i];
    }
}

for(int i = 1; i <= n; ++ i)
{
    if(ans1 > ans2) ++ ans2;
    else    ++ ans1;
}

for(int i = 1; i <= n; ++ i)
{
    if(ans1 > ans2) -- ans1;
    else    -- ans2;
}

cout<<min(ans1, ans2)<<endl;
```

[Problem - 1989B - Codeforces](https://codeforces.com/problemset/problem/1989/B)
$$
给出两个字符串a, b,求最短的s\\
s满足：a是s的子串，b是s的子序列\\
$$

$$
既然a是s的子串，则s中必然有一段是a，而s剩下的部分都是b的成分\\
故而需要找到一个a和b的重合部分c，有：\\
len(s)_{min}  = len(a) + len(b) - len(c)\\
我刚开始想的是c是a和b的最长公共子序列（样例居然还都过了\\
但实际上不是，例如：\\
string_a = ab, string_b = acb\\
然后很自然地考虑是否是最长公共子串，也不是:\\
string_a = acb, string_b = ab\\
好吧，那么c是满足这样的要求的：c既是a的子序列，又是b的子串\\
这个问题见动态规划部分
$$

```c++
string a, b;
cin>>a>>b;
int n = a.size(), m = b.size();

for(int i = 1; i <= n; ++ i)
    for(int j = 1; j <= m; ++ j)
         if(a[i - 1] == b[j - 1])
         	dp[i][j] = dp[i - 1][j - 1] + 1;
		else
            dp[i][j] = dp[i - 1][j];

int lenc = -INF;
for(int i = 1; i <= n; ++ i)
    for(int j = 1; j <= m; ++ j)
        lenc = max(ans, dp[i][j]);

cout<<a.size() + b.size() - lenc<<endl;
```

[Problem - 1987C - Codeforces](https://codeforces.com/problemset/problem/1987/C)
$$
每秒，若 i = n 或 a_i > a_{i + 1}，a_i减1（最多减到0）\\
问何时a中所有元素都减为0
$$

$$
对任意一个元素，总有：后面所有元素都变为0的下一秒，前面的元素才可能为0\\
当a[i]足够大时，即使后面的全都为0了，依然需要时间把a[i]变为0\\
记dp[i] 为：第i个元素变0的时刻\\
dp[i] = max(dp[i + 1] + 1, a[i])\\
$$

```c++
int a[MAXN] = {0}, dp[MAXN] = {0};
int n;
cin>>n;
for(int i = 1; i <= n; ++ i)
    cin>>a[i];

a[n + 1] = 0;
for(int i = n; i >= 1; -- i)
    dp[i] = max(dp[i + 1] + 1, a[i]);

cout<<dp[1];
```

[Problem - 1987B - Codeforces](https://codeforces.com/problemset/problem/1987/B)
$$
给定数组a，每次操作如下:\\
花费k + 1的代价，选出k个数，这k个数加1\\
问：使a变为单调不减时，花费的最小代价\\
$$

$$
既然要单调不减，那么a_i后面的所有数都要大于a_i\\
所以记录遍历到i时的最大值，记为ma\\
若a_i > ma, ma = a_i\\
否则把a_i补至ma，需要ma - a_i(记为cnt_i)个1\\
得到计数数组cnt，记max_{1 \leq i \leq n}(cnt_i) = cnt_m\\
那么，总的操作数就是cnt_m\\
一共要补\Sigma cnt，也就是S_{cnt}个1\\
\Sigma_{i = 1}^{cnt_m}(k_i + 1) = cnt_m + \Sigma_{i = 1}^{n}cnt_i\\
$$

```c++
int n;
cin>>n;
vector<int> cnt;
int a[MAXN] = {0};

for(int i = 1; i <= n; ++ i)
    cin>>a[i];
int ma = 0;

for(int i = 1; i <= n; ++ i)
{
    if(a[i] > ma)
        ma = a[i];
    else
        cnt.push_back(ma - a[i]);
}

int cntm = 0, ans = 0;
for(int i = 0; i < cnt.size(); ++ i)
{
    ans += cnt[i];
    cntm = max(cnt[i], cntm);
}
ans += cntm;

cout<<ans;
cout<<endl;
```

[Problem - 1986D - Codeforces](https://codeforces.com/problemset/problem/1986/D)
$$
给定一个数字构成的长度为n(2 \leq n \leq 20)的字符串s\\
插入n - 2个乘号或加号，求结果的最小值
$$

$$
暴力做法复杂度O(tn2^{n - 1}), 想一想贪心\\
n个数字有n - 1个空隙，插入n - 2个符号\\
相当于n个数字里选两个相邻数字，视为一个整体\\
注意到0和1：\\
0乘任何数都是0，所以尽可能拆成0和其它数相乘\\
1乘其它数不改变值，所以尽可能拆成1和其它数相乘\\
除去0、1之外的数，都相加而非相乘\\
至于合并哪两个数字，暴力求：\\
n个里面选连续的两个(也就是选一个),共n - 1种情况\\
复杂度O(n^2t)\\
$$

**buf**

```c++
int n;
cin>>n;
string s;
cin>>s;

int ans = INF;
for(int i = 0; i < s.size() - 1; ++ i)//枚举第i和i+1个数字合并后，算式的最小值
{
    //求第i个和i+1个数字合并之后的数组buf
    vector<int> buf;
    for(int j = 0; j < n; ++ j)
    {
        if(j != i && j != i + 1)
            buf.push_back( s[j] - '0');
        else
        {
            int num = (s[i] - '0') * 10 + s[i + 1] - '0';
            buf.push_back(num);
            j ++;//在原数组中占两位
        }
    }

    //下面求buf组成算式的最小值tmp
    int tmp = 0;
    for(int j = 0; j < buf.size(); ++ j)
    {
        //有一项为0，最小值就是都乘起来，答案是0
        if(buf[j] == 0)
        {
            cout<<0<<endl;
            return;
        }
        //乘1不改变值，忽略
        else if(buf[j] == 1)
            continue;
        //除了0和1外所有数都是相加
        else
            tmp += buf[j];
    }
    ans = min(ans, tmp);
    //注意到有特殊情况 ‘10’ 或 ‘101’ etc，执行到这一步时 ans 绝对不为 0，故对其与 1 取较大值即可
    ans = max((int)1, ans);
}
cout<<ans;
cout<<endl;
```

[Problem - E - Codeforces](https://codeforces.com/contest/1988/problem/E)
$$
单调栈\\
$$

```c++
/*
     * 伪代码
     * input
     * 求每个数对结果的贡献：
     * 用单调栈找到每个数影响的边界
     * 贪心：删除某个数后，找到的次小的数
     * 被删的数的贡献就是
     * /
```

[Problem - 1985F - Codeforces](https://codeforces.com/problemset/problem/1985/F)
$$
n种攻击，第i种伤害a_i，冷却c_i，boss血量h\\
问第几回合才能击败boss
$$

$$
由于n,h,a_i,c_i \leq 2e5，所以模拟会超时\\
求min\ t，使得：\\
\Sigma_{i = 1}^{n}[\frac{t - 1}{c_i} + 1] a_i \geq h \\
二分，t \in [1, 1e11]\\
复杂度O(nlog(nh))\\
此外要注意long \ long溢出
$$



[Problem - 1985G - Codeforces](https://codeforces.com/problemset/problem/1985/G)

[Problem - 1985E - Codeforces](https://codeforces.com/problemset/problem/1985/E)#

```c++
int x, y, z, k;
cin>>x>>y>>z>>k;
int ans = 0, ansi = 0;
for(int a = 1; a <= min(k, x); ++ a)
{
    if(k % a)
        continue;
    for(int b = 1; b <= min(k / a, y); ++ b)
    {
        int c = k / (a * b);
        if(a * b * c == k && c <= z)
        {
            ansi = (x - a + 1) * (y - b + 1) * (z - c + 1);
            ans = max(ansi, ans);
        }
    }
}
cout<<ans;
cout<<endl;
```

[CUGBACM23级暑假小学期训练 #18 - Virtual Judge (vjudge.net)](https://vjudge.net/contest/643437#problem/B)
$$
先排序a_i，记录a_i中出现的所有数，push\_back入b_i\\
cnt_i为b_i出现的次数
$$

```c++
clear(a);
vector<int> b, cnt;
int n, m;
cin>>n>>m;

for(int i = 1; i <= n; ++ i)
    cin>>a[i];

sort(a + 1, a + n + 1);

b.push_back(a[1]);
cnt.push_back( (int) 1 );
for(int i = 2; i <= n; ++ i)
{
    if(a[i] != b.back())
    {
        b.push_back( a[i] );
        cnt.push_back( (int)1 );
    }
    else
    {
        ++ cnt.back();
    }
}

int ans = 0;

//只有一个数时特判一下
if(b.size() == 1)
{
    int buf = cnt[0] * b[0];
    if(buf > m)
    {
        ans = max(ans, m / b[0] * b[0]);
    }
    else
    {
        ans = buf;
    }
    cout<<ans;
    cout<<endl;
    return;
}

//防止访问越界
b.push_back(m + 1);
cnt.push_back( (int)1 );

for(int j = 0; j < b.size() - 1; ++ j)
{
    for(int k = 0; k <= cnt[j]; ++ k)
    {
        int tmp = 0;
        int buf = k * b[j];
        if(buf > m)
            break;

        if(b[j] + 1 == b[j + 1])
            tmp = buf + min((m - buf) / b[j + 1], cnt[j + 1]) * b[j + 1];
        else
            tmp = buf;
        ans = max(ans, tmp);
    }
}
cout<<ans;
cout<<endl;
```



[Problem - B - Codeforces](https://codeforces.com/contest/1990/problem/B)
$$
构造一个1和-1的数列a，使得：\\
x为最小的最大前缀和下标，y为最大的最大后缀和下标\\
则有：x之后子段的前缀和小于等于0，y之前子段的后缀和大于等于0\\
$$




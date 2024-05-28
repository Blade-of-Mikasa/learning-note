## CF Round 944 div 4

#### A

```c++
//int数组：最大内存（MB）*1e6
#include <iostream>
#define INF 0x3f3f3f3f
#define ll long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define BEGIN 0//数组遍历起点
#define ARRAY2D(a)  {for(int i = BEGIN; i <= n; ++ i){cout<<"i="<<i<<" ";for(int j = BEGIN; j <= m; ++ j)cout<<std::right<<std::setw(3)<<a[i][j]<<' ';cout<<endl;}}//格式化输出二维数组
#define ARRAY1D(a)  {for(int i = BEGIN; i <= n; ++ i)cout<<std::right<<std::setw(3)<<a[i]<<' ';cout<<endl;}//格式化输出一维数组
using namespace std;
#define MAXN 105//数组大小
ll dp[MAXN], a[MAXN];

void solve()
{
    int a, b;
    cin>>a>>b;
    cout<<min(a, b)<<' '<<max(a, b)<<endl;
    return;
}


int main()
{
    ISO;
    int t;  cin>>t;
    while (t--)
    {
        solve();
    }
    //solve();
    return 0;
}

```



#### B

```c++
//int数组：最大内存（MB）*1e6
#include <iostream>
#define INF 0x3f3f3f3f
#define ll long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define BEGIN 0//数组遍历起点
#define ARRAY2D(a)  {for(int i = BEGIN; i <= n; ++ i){cout<<"i="<<i<<" ";for(int j = BEGIN; j <= m; ++ j)cout<<std::right<<std::setw(3)<<a[i][j]<<' ';cout<<endl;}}//格式化输出二维数组
#define ARRAY1D(a)  {for(int i = BEGIN; i <= n; ++ i)cout<<std::right<<std::setw(3)<<a[i]<<' ';cout<<endl;}//格式化输出一维数组
using namespace std;
#define MAXN 105//数组大小
ll dp[MAXN], a[MAXN];

void solve()
{
    string s;   cin>>s;
    int size = s.size();
    if(size == 1)
    {
        cout<<"NO"<<endl;
        return;
    }
    for(int i = 0; i < size; ++ i)
    {
        for(int j = i; j <size; ++ j)
        {
            if(s[i] != s[j])
            {
                swap(s[i], s[j]);
                cout<<"YES"<<endl;
                cout<<s<<endl;
                return;
            }
        }
    }
    cout<<"NO"<<endl;
    return;
}


int main()
{
    ISO;
    int t;  cin>>t;
    while (t--)
    {
        solve();
    }
    //solve();
    return 0;
}

```

#### C

```c++
//int数组：最大内存（MB）*1e6
#include <iostream>
#define INF 0x3f3f3f3f
#define ll long long
#define ISO  {std::ios::sync_with_stdio(false);cin.tie(0); cout.tie(0);}//关流
#define BEGIN 0//数组遍历起点
#define ARRAY2D(a)  {for(int i = BEGIN; i <= n; ++ i){cout<<"i="<<i<<" ";for(int j = BEGIN; j <= m; ++ j)cout<<std::right<<std::setw(3)<<a[i][j]<<' ';cout<<endl;}}//格式化输出二维数组
#define ARRAY1D(a)  {for(int i = BEGIN; i <= n; ++ i)cout<<std::right<<std::setw(3)<<a[i]<<' ';cout<<endl;}//格式化输出一维数组
using namespace std;
#define MAXN 105//数组大小
ll dp[MAXN];
int in(int a, int z, int y)
{
    int l = min(z, y);
    int r = max(z, y);
    if(a > l && a < r)
        return 1;
    else
        return -1;
}

void solve()
{
    int a, b, c, d; cin>>a>>b>>c>>d;
    if(in(c, a, b) * in(d, a, b) == -1)
    {
        cout<<"yes"<<endl;
    }
    else
        cout<<"no"<<endl;
    return;
}


int main()
{
    ISO;
    int t;  cin>>t;
    while (t--)
    {
        solve();
    }
    //solve();
    return 0;
}

```


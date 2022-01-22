# DP
for i in range(N):
    w, v = weight[i], value[i]
    for j in range(W + 1):
        if B[i] <= w:
            dp[i + 1][j] = max(dp[i][j - w] +
                               v, dp[i][j])
        else:  # 入る可能性はない
            dp[i + 1][j] = dp[i][j]
print(dp[N][W])

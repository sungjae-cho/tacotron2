def decompose_hangul(text):
    Start_Code, ChoSung, JungSung = 44032, 588, 28
    ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
    JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                     'ㅣ']
    JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                     'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

    line_dec = ""
    line = list(text.strip())

    for keyword in line:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            char_code = ord(keyword) - Start_Code
            char1 = int(char_code / ChoSung)
            line_dec += ChoSung_LIST[char1]
            char2 = int((char_code - (ChoSung * char1)) / JungSung)
            line_dec += JungSung_LIST[char2]
            char3 = int((char_code - (ChoSung * char1) - (JungSung * char2)))
            line_dec += JongSung_LIST[char3]
        else:
            line_dec += keyword
    return line_dec


'''
Reference: https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py
'''
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

min_duration = 1 # seconds
max_duration = 10
hard_duration_mask = (y > min_duration) & (y < max_duration)
i_hard_duration = np.argwhere(hard_duration_mask)
i_out_duration = np.argwhere(np.invert(hard_duration_mask))

x_in_range = x[i_hard_duration]
y_in_range = y[i_hard_duration]
x_out_range = x[i_out_duration]
y_out_range = y[i_out_duration]

# Create linear regression object
regr = linear_model.LinearRegression(fit_intercept=False)

# Train the model using the training sets
regr.fit(x_in_range, y_in_range)

# Make predictions using the training set
y_pred = regr.predict(x.reshape(-1, 1))
y_pred = y_pred.reshape(-1)

# The coefficients
print('Coefficients: \n', regr.coef_,)
# The intercpet
print('Intercept: \n', regr.intercept_)
# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_in_range, y_pred[i_hard_duration]))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_in_range, y_pred[i_hard_duration]))

# Plot outputs
plt.title("Audios Segmented by Duration in Range ({}, {})".format(min_duration, max_duration))
plt.scatter(x_in_range, y_in_range, color='green', marker='.', alpha=0.05)
plt.scatter(x_out_range, y_out_range, color='red', marker='.', alpha=0.05)
plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xticks([x_in_range.min(), x_in_range.max()])
plt.yticks([y_in_range.min(), y_in_range.max()])
plt.xlabel('len(phone_seq)')
plt.ylabel('Duration(sec)')

plt.show()

print("Audios between {} and {} secs: {}".format(min_duration, max_duration, hard_duration_mask.sum()))
print("Audios out of duration between {} and {} secs: {}".format(min_duration, max_duration, np.invert(hard_duration_mask).sum()))

print("Total samples:", len(x))
print("Inliers:", len(x_in_range))
secs_inliers = int(y_in_range.sum())
h, m, s = convert(secs_inliers)
print("Duration(inliers): {}h {}m {}s".format(h, m, s))

print("Outliers:", len(x_out_range))
secs_outliers = int(y_out_range.sum())
h, m, s = convert(secs_outliers)
print("Duration(outliers): {}h {}m {}s".format(h, m, s))

print("Ratio of Outliers:", (len(x_out_range) / len(x)))

# AE: Absolute Error between durations and predicted durations
ae_y = np.abs((y - y_pred))
print(ae_y.shape)

iqr_ae_y = scipy.stats.iqr(ae_y)
q1 = np.quantile(ae_y, 0.25)
q3 = np.quantile(ae_y, 0.75)
lb_ae_y = q1 - 1.5 * iqr_ae_y # Lower bound
ub_ae_y = q3 + 1.5 * iqr_ae_y # Upper bound
print("The lower bound of 1.5*IQR outlying duration absolute errors:", lb_ae_y)
print("The upper bound of 1.5*IQR outlying duration absolute errors:", ub_ae_y)

outlier_mask = (((ae_y < lb_ae_y) | (ae_y > ub_ae_y)) & hard_duration_mask)
i_outliers = np.argwhere(outlier_mask).reshape(-1)
inlier_mask = ((np.invert((ae_y < lb_ae_y) | (ae_y > ub_ae_y))) & hard_duration_mask)
i_inliers = np.argwhere(inlier_mask).reshape(-1)

x_outliers = x[i_outliers]
y_outliers = y[i_outliers]
x_inliers = x[i_inliers]
y_inliers = y[i_inliers]

plt.title('1.5 IQR Outliers by Duration Absolute Errors from Linear Model')

# Plot outputs
plt.scatter(x_outliers, y_outliers, color='red', marker='.', alpha=0.05)
plt.scatter(x_inliers, y_inliers, color='green', marker='.', alpha=0.05)


plt.plot(x, y_pred, color='blue', linewidth=3)

plt.xticks([x_inliers.min(), x_inliers.mean(), x_inliers.max(), x.min(), x.max()])
plt.yticks([y_inliers.min(), y_inliers.mean(), y_inliers.max(), y.min(), y.max()])
plt.xlabel('len(phone_seq)')
plt.ylabel('Duration(sec)')

plt.show()

print("Total samples:", (len(x_outliers) + len(x_inliers)))
print("Inliers:", len(x_inliers))
secs_inliers = int(y[i_inliers].sum())
h, m, s = convert(secs_inliers)
print("Duration(inliers): {}h {}m {}s".format(h, m, s))

print("Outliers:", len(x_outliers))
secs_outliers = int(y[i_outliers].sum())
h, m, s = convert(secs_outliers)
print("Duration(outliers): {}h {}m {}s".format(h, m, s))

print("Ratio of Outliers:", (len(x_outliers) / (len(x_outliers) + len(x_inliers))))
def months_to_years(df, column_name, year_title = 'years'):
    df[year_title] = [int(c / 12) for c in df[column_name]]


print('<<<<<>>>>>')
train, test = train_test_split(telco, train_size = .80, random_state = 123)
print(train)
print(test)
scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(train)


train_scaled_data = scaler.transform(train)
test_scaled_data = scaler.transform(test)

# transform train
train_scaled = pd.DataFrame(scaler.transform(train), columns=train.columns.values).set_index([train.index.values])
# transform test
test_scaled = pd.DataFrame(scaler.transform(test), columns=test.columns.values).set_index([test.index.values])

train_unscaled = pd.DataFrame(scaler.inverse_transform(train_scaled), columns=train_scaled.columns.values).set_index([train.index.values])
test_unscaled = pd.DataFrame(scaler.inverse_transform(test_scaled), columns=test_scaled.columns.values).set_index([test.index.values])



print(train_scaled)
print(test_scaled)
print(train_scaled)
sns.jointplot(x="total_charges", y="monthly_charges", data=train_scaled, kind="reg");
plt.show()
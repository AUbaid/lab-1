def print_reg_perf(y_preds, y_actuals, set_name=None):

    from sklearn.metrics import mean_squared_error as mse
    from sklearn.metrics import mean_absolute_error as mae

    print(f"RMSE {set_name}: {mse(y_actuals, y_preds, squared=False)}")
    print(f"MAE {set_name}: {mae(y_actuals, y_preds)}")
def expected_calibration_error(y_true, y_pred, n_bins):
    """
    Compute Expected Calibration Error.
    """
    yn=len(y_true)
    yp=len(y_pred)
    
    acc=0
    conf=0
    ece=0

    bin_size = 1.0 / n_bins

    for b in range(n_bins):
        bin_start = b * bin_size
        bin_end = (b + 1) * bin_size

        curr_acc = 0
        curr_conf = 0
        count = 0

        for i in range(yn):
            if y_pred[i] >= bin_start and y_pred[i] < bin_end:
                count += 1
                curr_acc += y_true[i]
                curr_conf += y_pred[i]

        if count > 0:
            acc = curr_acc / count
            conf = curr_conf / count
            ece += (count / yn) * abs(acc - conf)

    return ece
def standard_bolus(CF, ICR, CHO, G, Gt):
    CHO = CHO * 1e-3                                            # convert CHO from mg to g
    u = CHO / ICR + (G - Gt) / CF                               # compute bolus in U
    u = round(2*u) / 2                                          # round to 0.5U steps
    u = max(0, u)                                               # u >= 0
    return u*1e6                                                # convert bolus to muU

def UKF_bolus(IS, ISb, CF, ICR, CHO, G, Gt):
    CHO = CHO * 1e-3                                            # convert CHO from mg to g
    u = ISb / IS * (CHO / ICR + (G - Gt) / CF)                  # compute bolus in U
    u = round(2*u) / 2                                          # round to 0.5U steps
    u = max(0, u)                                               # u >= 0
    return u*1e6                                                # convert bolus to muU

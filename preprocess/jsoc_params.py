class jsocParams():
    def __init__(self, instr="hmi"):
        if instr=="hmi":
            self.series_name = "hmi.V_sht_2drls"
            self.daylist_fname = "daylist.hmi"
            self.LMAX = "200"
            self.NDT = "138240"
            self.freq_series_name = "hmi.v_sht_modes"

        elif instr=="mdi":
            self.series_name = "mdi.vw_V_sht_2drls"
            self.daylist_fname = "daylist.mdi"
            self.LMAX = "300"
            self.NDT = "103680"
            self.freq_series_name = "mdi.vw_V_sht_modes"

        elif instr=="mdi-360d":
            self.series_name = "mdi.vw_V_sht_2drls"
            self.daylist_fname = "daylist.mdi"
            self.LMAX = "300"
            self.NDT = "518400"
            self.freq_series_name = "mdi.vw_V_sht_modes"



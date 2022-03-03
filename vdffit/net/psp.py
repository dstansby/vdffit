from sunpy.net import Fido, attrs as a


def fetch_psp_files(starttime, endtime):
    t = a.Time(starttime, endtime)
    datasets = ['PSP_SWP_SPI_SF00_L2_8DX32EX8A',
                'PSP_FLD_L2_MAG_SC_4_SA_PER_CYC']
    for dataset in datasets:
        result = Fido.search(t, a.cdaweb.Dataset(dataset))
        print(result)
        Fido.fetch(result)

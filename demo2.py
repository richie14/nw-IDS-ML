from flowmeter import Flowmeter

feature_gen = Flowmeter(
    offline = "testbed-13jun.pcap",
    outfunc = None,
    outfile = "output.csv")

feature_gen.run()


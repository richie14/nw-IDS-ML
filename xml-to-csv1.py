# Importing the required libraries
import xml.etree.ElementTree as Xet
import pandas as pd

cols = ["appName", "totalSourceBytes", "totalDestinationBytes", "totalDestinationPackets", "totalSourcePackets", "sourcePayloadAsBase64", "destinationPayloadAsBase64", "destinationPayloadAsUTF", "direction", "sourceTCPFlagsDescription", "destinationTCPFlagsDescription", "source", "protocolName", "sourcePort", "destination", "destinationPort","startDateTime","stopDateTime","Tag"]
rows = []

# Parsing the XML file
xmlparse = Xet.parse('TestbedSatJun12Flows.xml')
root = xmlparse.getroot()
for i in root:
	appName = i.find("appName").text
	tSrcB = i.find("totalSourceBytes").text
	tDstB = i.find("totalDestinationBytes").text
	tDstP = i.find("totalDestinationPackets").text
	tSrcP = i.find("totalSourcePackets").text
	srcB64 = i.find("sourcePayloadAsBase64").text
	dstB64 = i.find("destinationPayloadAsBase64").text
	dstUTF = i.find("destinationPayloadAsUTF").text
	dirtn = i.find("direction").text
	srcTCPF = i.find("sourceTCPFlagsDescription").text
	dstTCPF = i.find("destinationTCPFlagsDescription").text
	src = i.find("source").text
	protocol = i.find("protocolName").text
	srcPt = i.find("sourcePort").text
	dst = i.find("destination").text
	dstPt = i.find("destinationPort").text
	startDT = i.find("startDateTime").text
	stopDT = i.find("stopDateTime").text
	tag = i.find("Tag").text
	
	
	 

	rows.append({"appName": appName,
				"totalSourceBytes": tSrcB,
				"totalDestinationBytes": tDstB,
				"totalDestinationPackets": tDstP,
				"totalSourcePackets": tSrcP,
				"sourcePayloadAsBase64": srcB64,
				"destinationPayloadAsBase64": dstB64,
				"destinationPayloadAsUTF": dstUTF,
				"direction": dirtn,
				"sourceTCPFlagsDescription": srcTCPF,
				"destinationTCPFlagsDescription": dstTCPF ,	
				"source": src,
				"protocolName": protocol,	
				"sourcePort": srcPt,
				"destination": dst,	
				"destinationPort": dstPt,
				"startDateTime": startDT,	
				"stopDateTime": stopDT,
				"Tag": tag })

df = pd.DataFrame(rows, columns=cols)
df = df.replace(';','richa', regex=True)


# Writing dataframe to csv
df.to_csv('Jun17output1.csv')


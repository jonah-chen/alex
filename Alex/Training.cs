﻿using System;
using System.Collections.Generic;
using System.Text;
using YahooFinanceApi;
using MathNet.Numerics.LinearAlgebra;
using System.Threading.Tasks;


namespace Alex
{
    class Training_Instance
    {
        public double target;
        public List<double> inputs = new List<double>();
        public string tinker;
        public DateTime dt;
        public Training_Instance(string tinker, DateTime dt)
        {
            this.tinker = tinker;
            this.dt = dt;
        }
    }
    public static class Training
    {
        public static readonly string[] DATA =
        {
           /*"A",
"AAL",
"AAP",*/
"AAPL"/*,
"ABBV",
"ABC",
"ABMD",
"ABT",
"ACN",
"ADBE",
"ADI",
"ADM",
"ADP",
"ADS",
"ADSK",
"AEE",
"AEP",
"AES",
"AFL",
"AGN",
"AIG",
"AIV",
"AIZ",
"AJG",
"AKAM",
"ALB",
"ALGN",
"ALK",
"ALL",
"ALLE",
"ALXN",
"AMAT",
"AMCR",
"AMD",
"AME",
"AMG",
"AMGN",
"AMP",
"AMT",
"AMZN",
"ANET",
"ANSS",
"ANTM",
"AON",
"AOS",
"APA",
"APD",
"APH",
"APTV",
"ARE",
"ATO",
"ATVI",
"AVB",
"AVGO",
"AVY",
"AWK",
"AXP",
"AZO",
"BA",
"BAC",
"BAX",
"BBY",
"BDX",
"BEN",
"BIIB",
"BK",
"BKNG",
"BLK",
"BLL",
"BMY",
"BR",
"BSX",
"BWA",
"BXP",
"C",
"CAG",
"CAH",
"CAT",
"CB",
"CBOE",
"CBRE",
"CBS",
"CCI",
"CCL",
"CDNS",
"CE",
"CELG",
"CERN",
"CF",
"CFG",
"CHD",
"CHRW",
"CHTR",
"CI",
"CINF",
"CL",
"CLX",
"CMA",
"CMCSA",
"CME",
"CMG",
"CMI",
"CMS",
"CNC",
"CNP",
"COF",
"COG",
"COO",
"COP",
"COST",
"COTY",
"CPB",
"CPRI",
"CPRT",
"CRM",
"CSCO",
"CSX",
"CTAS",
"CTL",
"CTSH",
"CTXS",
"CVS",
"CVX",
"CXO",
"D",
"DAL",
"DD",
"DE",
"DFS",
"DG",
"DGX",
"DHI",
"DHR",
"DIS",
"DISCA",
"DISCK",
"DISH",
"DLR",
"DLTR",
"DOV",
"DRE",
"DRI",
"DTE",
"DUK",
"DVA",
"DVN",
"DXC",
"EA",
"EBAY",
"ECL",
"ED",
"EFX",
"EIX",
"EL",
"EMN",
"EMR",
"EOG",
"EQIX",
"EQR",
"ES",
"ESS",
"ETFC",
"ETN",
"ETR",
"EVRG",
"EW",
"EXC",
"EXPD",
"EXPE",
"EXR",
"F",
"FANG",
"FAST",
"FB",
"FBHS",
"FCX",
"FDX",
"FE",
"FFIV",
"FIS",
"FISV",
"FITB",
"FLIR",
"FLS",
"FLT",
"FMC", 
"FRC",
"FRT",
"FTI",
"FTNT",
"FTV",
"GD",
"GE",
"GILD",
"GIS",
"GL",
"GLW",
"GM",
"GOOG",
"GOOGL",
"GPC",
"GPN",
"GPS",
"GRMN",
"GS",
"GWW",
"HAL",
"HAS",
"HBAN",
"HBI",
"HCA",
"HD",
"HES",
"HFC",
"HIG",
"HII",
"HLT",
"HOG",
"HOLX",
"HON",
"HP",
"HPE",
"HPQ",
"HRB",
"HRL",
"HSIC",
"HST",
"HSY",
"HUM",
"IBM",
"ICE",
"IDXX",
"IEX",
"IFF",
"ILMN",
"INCY",
"INFO",
"INTC",
"INTU",
"IP",
"IPG",
"IPGP",
"IQV",
"IR",
"IRM",
"ISRG",
"IT",
"ITW",
"IVZ",
"JBHT",
"JCI",
"JEC",
"JEF",
"JKHY",
"JNJ",
"JNPR",
"JPM",
"JWN",
"K",
"KEY",
"KEYS",
"KHC",
"KIM",
"KLAC",
"KMB",
"KMI",
"KMX",
"KO",
"KR",
"KSS",
"KSU",
"L",
"LB",
"LDOS",
"LEG",
"LEN",
"LH",
"LHX",
"LIN",
"LKQ",
"LLY",
"LMT",
"LNC",
"LNT",
"LOW",
"LRCX",
"LUV",
"LW",
"LYB",
"M",
"MA",
"MAA",
"MAC",
"MAR",
"MAS",
"MCD",
"MCHP",
"MCK",
"MCO",
"MDLZ",
"MDT",
"MET",
"MGM",
"MHK",
"MKC",
"MKTX",
"MLM",
"MMC",
"MMM",
"MNST",
"MO",
"MOS",
"MPC",
"MRK",
"MRO",
"MS",
"MSCI",
"MSFT",
"MSI",
"MTB",
"MTD",
"MU",
"MXIM",
"MYL",
"NBL",
"NCLH",
"NDAQ",
"NEE",
"NEM",
"NFLX",
"NI",
"NKE",
"NKTR",
"NLSN",
"NOC",
"NOV",
"NRG",
"NSC",
"NTAP",
"NTRS",
"NUE",
"NVDA",
"NWL",
"NWS",
"NWSA",
"O",
"OKE",
"OMC",
"ORCL",
"ORLY",
"OXY",
"PAYX",
"PBCT",
"PCAR",
"PEG",
"PEP",
"PFE",
"PFG",
"PG",
"PGR",
"PH",
"PHM",
"PKG",
"PKI",
"PLD",
"PM",
"PNC",
"PNR",
"PNW",
"PPG",
"PPL",
"PRGO",
"PRU",
"PSA",
"PSX",
"PVH",
"PWR",
"PXD",
"PYPL",
"QCOM",
"QRVO",
"RCL",
"RE",
"REG",
"REGN",
"RF",
"RHI",
"RJF",
"RL",
"RMD",
"ROK",
"ROL",
"ROP",
"ROST",
"RSG",
"RTN",
"SBAC",
"SBUX",
"SCHW",
"SEE",
"SHW",
"SIVB",
"SJM",
"SLB",
"SLG",
"SNA",
"SNPS",
"SO",
"SPG",
"SPGI",
"SRE",
"STI",
"STT",
"STX",
"STZ",
"SWK",
"SWKS",
"SYF",
"SYK",
"SYY",
"T",
"TAP",
"TDG",
"TEL",
"TFX",
"TGT",
"TIF",
"TJX",
"TMO",
"TMUS",
"TPR",
"TRIP",
"TROW",
"TRV",
"TSCO",
"TSN",
"TTWO",
"TWTR",
"TXN",
"TXT",
"UA",
"UAA",
"UAL",
"UDR",
"UHS",
"ULTA",
"UNH",
"UNM",
"UNP",
"UPS",
"URI",
"USB",
"UTX",
"V",
"VAR",
"VFC",
"VIAB",
"VLO",
"VMC",
"VNO",
"VRSK",
"VRSN",
"VRTX",
"VTR",
"VZ",
"WAB",
"WAT",
"WBA",
"WCG",
"WDC",
"WEC",
"WELL",
"WFC",
"WHR",
"WLTW",
"WM",
"WMB",
"WMT",
"WRK",
"WU",
"WY",
"WYNN",
"XEC",
"XEL",
"XLNX",
"XOM",
"XRAY",
"XRX",
"XYL",
"YUM",
"ZBH",
"ZION",
"ZTS"*/
        };

        public static double ReLU(double x)
        {
            return Math.Max(0, x);
        }
        public static double derivReLU(double x)
        {
            if (x <= 0.0)
                return 0.0;
            return 1.0;
        }
        public static Vector<double> derivReLU(Vector<double> x)
        {
            Vector<double> y = x;
            for(int i = 0; i < x.Count; i++)
            {
                if (x[i] <= 0.0)
                {
                    y[i] = 0.0;
                }
                else
                    y[i] = 1.0;
            }
            return y;
        }
        public static double cost (double output, double target)
        {
            return (output - target) * (output - target) * 0.5; 
        }
        public static double derivCost(double output, double target)
        {
            return output - target;
        }

        
    }
}

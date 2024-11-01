"""A relatively large java codeforces example that contained the 'wide' keyword"""

import os
from bincfg import CFG, CFGBasicBlock, CFGFunction, CFGEdge, EdgeType
from ..fake_classes import FakeCFG, FakeCFGFunction


def get_manual_cfg(build_level):
    """Returns a manually built control flow graph
    
    Args:
        build_level (str): the level at which to build the cfg. Can be:

            - 'cfg': will build the full CFG() object
            - 'function': will build full CFGFunction's, but use a FakeCFG() object for the CFG
            - 'block': will build only CFGBasicBlock's. A FakeCFGFunction() object will be built as the functions that is
              simply an empty data structure to hold all the construction attributes being used, but with no processing

    Returns:
        dict[str, Any]: dictionary with the following keys/values:

            - 'blocks' (dict[int, CFGBasicBlock]): dictionary mapping basic block addresses to basic blocks
            - 'functions' (Union[dict[int, CFGFunction], dict[int, FakeCFGFunction]]): dictionary mapping function addresses
              to CFGFunctions (or, FakeCFGFunctions if we are not building functions)
            - 'cfg' (Union[CFG, object]): the CFG() object (if make_cfg=True) else a new object()
            - 'inputs' (list[CFGInputDataType]): input values that, when passed into CFG() constructor, should produce
              the exact same CFG() as that in 'cfg' (when build_level='cfg')
            - 'expected' (dict[str, Any]): dictionary of expected values. The following values are present:

                * 'sorted_block_order' (list[int]): list of basic block addresses in sorted order
                * 'sorted_func_order' (list[int]): list of function addresses in sorted order
                * 'num_blocks' (dict[int, int]): number of blocks per function, keys are function addresses
                * 'num_asm_lines_per_block' (dict[int, int]): number of asm lines per block, keys are block addresses
                * 'num_asm_lines_per_function' (dict[int, int]): number of asm lines per function, keys are function addresses
                * 'num_functions' (int): the number of functions
                * 'is_root_function' (dict[int, bool]): True if the function is a root function, keys are function addresses
                * 'is_extern_function' (dict[int, bool]): True if the function is an external function, keys are function addresses
                * 'is_intern_function' (dict[int, bool]): True if the function is an internal function, keys are function addresses
                * 'function_entry_block' (dict[int, int]): the address of the function entry block for each function, keys are function addresses
                * 'called_by' (dict[int, set[int]]): set of addresses of basic blocks that call each function, keys are function addresses
                * 'asm_counts_per_block' (dict[int, dict[str, int]]): dictionary of assembly line counts for each block, keys are block addresses
                * 'asm_counts_per_function' (dict[int, dict[str, int]]): dictionary of assembly line counts for each function, keys are function addresses
                * 'asm_counts' (dict[str, int]): dictinary of assembly line counts for entire CFG
    """
    if build_level not in ['cfg', 'function', 'block']:
        raise ValueError("Bad build_level: %s" % repr(build_level))
    
    func_type = CFGFunction if build_level in ['function', 'cfg'] else FakeCFGFunction

    # Create the cfg object. This cfg has 30 functions, 250 basic blocks, 359 edges, and 844 lines of assembly.
    __auto_cfg = CFG() if build_level in ['cfg'] else FakeCFG()

    __auto_functions = {
        902: func_type(parent_cfg=__auto_cfg, address=902, name='CF::<init>', is_extern_function=False, metadata={}),
        945: func_type(parent_cfg=__auto_cfg, address=945, name='CF::main', is_extern_function=False, metadata={}),
        1295: func_type(parent_cfg=__auto_cfg, address=1295, name='CF::Permute', is_extern_function=False, metadata={}),
        1534: func_type(parent_cfg=__auto_cfg, address=1534, name='CF::radixSort', is_extern_function=False, metadata={}),
        1579: func_type(parent_cfg=__auto_cfg, address=1579, name='CF::radixSort', is_extern_function=False, metadata={}),
        2001: func_type(parent_cfg=__auto_cfg, address=2001, name='CF::<clinit>', is_extern_function=False, metadata={}),
        5216: func_type(parent_cfg=__auto_cfg, address=5216, name='CF$FastScanner::<init>', is_extern_function=False, metadata={}),
        5285: func_type(parent_cfg=__auto_cfg, address=5285, name='CF$FastScanner::read', is_extern_function=False, metadata={}),
        5471: func_type(parent_cfg=__auto_cfg, address=5471, name='CF$FastScanner::isSpaceChar', is_extern_function=False, metadata={}),
        5556: func_type(parent_cfg=__auto_cfg, address=5556, name='CF$FastScanner::isEndline', is_extern_function=False, metadata={}),
        5629: func_type(parent_cfg=__auto_cfg, address=5629, name='CF$FastScanner::nextInt', is_extern_function=False, metadata={}),
        5675: func_type(parent_cfg=__auto_cfg, address=5675, name='CF$FastScanner::nextArrayInt', is_extern_function=False, metadata={}),
        5773: func_type(parent_cfg=__auto_cfg, address=5773, name='CF$FastScanner::nextArrayString', is_extern_function=False, metadata={}),
        5872: func_type(parent_cfg=__auto_cfg, address=5872, name='CF$FastScanner::nextLong', is_extern_function=False, metadata={}),
        5918: func_type(parent_cfg=__auto_cfg, address=5918, name='CF$FastScanner::nextDouble', is_extern_function=False, metadata={}),
        5964: func_type(parent_cfg=__auto_cfg, address=5964, name='CF$FastScanner::next', is_extern_function=False, metadata={}),
        6102: func_type(parent_cfg=__auto_cfg, address=6102, name='CF$FastScanner::nextLine', is_extern_function=False, metadata={}),
        18446744073709508607: func_type(parent_cfg=__auto_cfg, address=18446744073709508607, name='java/lang/StringBuilder::toString', is_extern_function=False, metadata={}),
        18446744073709511679: func_type(parent_cfg=__auto_cfg, address=18446744073709511679, name='java/lang/StringBuilder::appendCodePoint', is_extern_function=False, metadata={}),
        18446744073709512703: func_type(parent_cfg=__auto_cfg, address=18446744073709512703, name='java/lang/StringBuilder::<init>', is_extern_function=False, metadata={}),
        18446744073709524991: func_type(parent_cfg=__auto_cfg, address=18446744073709524991, name='java/lang/Double::parseDouble', is_extern_function=False, metadata={}),
        18446744073709527039: func_type(parent_cfg=__auto_cfg, address=18446744073709527039, name='java/lang/Long::parseLong', is_extern_function=False, metadata={}),
        18446744073709531135: func_type(parent_cfg=__auto_cfg, address=18446744073709531135, name='java/lang/Integer::parseInt', is_extern_function=False, metadata={}),
        18446744073709533183: func_type(parent_cfg=__auto_cfg, address=18446744073709533183, name='java/util/InputMismatchException::<init>', is_extern_function=False, metadata={}),
        18446744073709534207: func_type(parent_cfg=__auto_cfg, address=18446744073709534207, name='java/io/InputStream::read', is_extern_function=False, metadata={}),
        18446744073709536255: func_type(parent_cfg=__auto_cfg, address=18446744073709536255, name='java/lang/Object::<init>', is_extern_function=False, metadata={}),
        18446744073709537279: func_type(parent_cfg=__auto_cfg, address=18446744073709537279, name='java/util/LinkedList::<init>', is_extern_function=False, metadata={}),
        18446744073709540351: func_type(parent_cfg=__auto_cfg, address=18446744073709540351, name='java/util/List::add', is_extern_function=False, metadata={}),
        18446744073709541375: func_type(parent_cfg=__auto_cfg, address=18446744073709541375, name='java/io/PrintStream::println', is_extern_function=False, metadata={}),
        18446744073709542399: func_type(parent_cfg=__auto_cfg, address=18446744073709542399, name='java/lang/Math::min', is_extern_function=False, metadata={}),
        18446744073709544447: func_type(parent_cfg=__auto_cfg, address=18446744073709544447, name='java/lang/Math::abs', is_extern_function=False, metadata={}),
    }

    # Building basic blocks. Dictionary maps integer address to CFGBasicBlock() object
    __auto_blocks = {
        902: CFGBasicBlock(parent_function=__auto_functions[902], address=902, asm_memory_addresses=[902, 903], metadata={}, asm_lines=[
            'aload_0',
            'invokespecial 0x0001<java/lang/Object::<init>>',
        ]),
        906: CFGBasicBlock(parent_function=__auto_functions[902], address=906, asm_memory_addresses=[906], metadata={}, asm_lines=[
            'return',
        ]),
        945: CFGBasicBlock(parent_function=__auto_functions[945], address=945, asm_memory_addresses=[945, 948, 949, 952], metadata={}, asm_lines=[
            'new    0x0007<CF$FastScanner>',
            'dup',
            'getstatic 0x0009<java/lang/System::in>',
            'invokespecial 0x000f<CF$FastScanner::<init>>',
        ]),
        955: CFGBasicBlock(parent_function=__auto_functions[945], address=955, asm_memory_addresses=[955, 956, 957], metadata={}, asm_lines=[
            'astore_1',
            'aload_1',
            'invokevirtual 0x0012<CF$FastScanner::nextInt>',
        ]),
        960: CFGBasicBlock(parent_function=__auto_functions[945], address=960, asm_memory_addresses=[960, 961, 962], metadata={}, asm_lines=[
            'istore_2',
            'aload_1',
            'invokevirtual 0x0012<CF$FastScanner::nextInt>',
        ]),
        965: CFGBasicBlock(parent_function=__auto_functions[945], address=965, asm_memory_addresses=[965, 966, 967], metadata={}, asm_lines=[
            'istore_3',
            'aload_1',
            'invokevirtual 0x0012<CF$FastScanner::nextInt>',
        ]),
        970: CFGBasicBlock(parent_function=__auto_functions[945], address=970, asm_memory_addresses=[970, 972, 973], metadata={}, asm_lines=[
            'istore 0x04',
            'aload_1',
            'invokevirtual 0x0012<CF$FastScanner::nextInt>',
        ]),
        976: CFGBasicBlock(parent_function=__auto_functions[945], address=976, asm_memory_addresses=[976, 978, 979, 981, 983, 984], metadata={}, asm_lines=[
            'istore 0x05',
            'iconst_0',
            'istore 0x06',
            'iload  0x04',
            'iconst_1',
            'if_icmple 0x0028',
        ]),
        987: CFGBasicBlock(parent_function=__auto_functions[945], address=987, asm_memory_addresses=[987, 989, 990, 992, 993], metadata={}, asm_lines=[
            'iload  0x06',
            'iload_3',
            'iload  0x04',
            'isub',
            'invokestatic 0x0016<java/lang/Math::abs>',
        ]),
        996: CFGBasicBlock(parent_function=__auto_functions[945], address=996, asm_memory_addresses=[996, 997, 999, 1002, 1004, 1005], metadata={}, asm_lines=[
            'iadd',
            'istore 0x06',
            'iinc   0x06, 0x01',
            'iload  0x05',
            'iload_2',
            'if_icmpge 0x0019',
        ]),
        1008: CFGBasicBlock(parent_function=__auto_functions[945], address=1008, asm_memory_addresses=[1008, 1010, 1012, 1014, 1015, 1016, 1018, 1021], metadata={}, asm_lines=[
            'iload  0x06',
            'iload  0x05',
            'iload  0x04',
            'isub',
            'iadd',
            'istore 0x06',
            'iinc   0x06, 0x01',
            'goto   0x0009',
        ]),
        1024: CFGBasicBlock(parent_function=__auto_functions[945], address=1024, asm_memory_addresses=[1024], metadata={}, asm_lines=[
            'wide   0x84<-124>, 0x0006, 0x03e8<1000>',
        ]),
        1030: CFGBasicBlock(parent_function=__auto_functions[945], address=1030, asm_memory_addresses=[1030, 1031, 1033, 1035, 1036], metadata={}, asm_lines=[
            'iconst_0',
            'istore 0x07',
            'iload  0x05',
            'iload_2',
            'if_icmpge 0x0028',
        ]),
        1039: CFGBasicBlock(parent_function=__auto_functions[945], address=1039, asm_memory_addresses=[1039, 1041, 1042, 1044, 1045], metadata={}, asm_lines=[
            'iload  0x07',
            'iload_3',
            'iload  0x05',
            'isub',
            'invokestatic 0x0016<java/lang/Math::abs>',
        ]),
        1048: CFGBasicBlock(parent_function=__auto_functions[945], address=1048, asm_memory_addresses=[1048, 1049, 1051, 1054, 1056, 1057], metadata={}, asm_lines=[
            'iadd',
            'istore 0x07',
            'iinc   0x07, 0x01',
            'iload  0x04',
            'iconst_1',
            'if_icmple 0x0019',
        ]),
        1060: CFGBasicBlock(parent_function=__auto_functions[945], address=1060, asm_memory_addresses=[1060, 1062, 1064, 1066, 1067, 1068, 1070, 1073], metadata={}, asm_lines=[
            'iload  0x07',
            'iload  0x05',
            'iload  0x04',
            'isub',
            'iadd',
            'istore 0x07',
            'iinc   0x07, 0x01',
            'goto   0x0009',
        ]),
        1076: CFGBasicBlock(parent_function=__auto_functions[945], address=1076, asm_memory_addresses=[1076], metadata={}, asm_lines=[
            'wide   0x84<-124>, 0x0007, 0x03e8<1000>',
        ]),
        1082: CFGBasicBlock(parent_function=__auto_functions[945], address=1082, asm_memory_addresses=[1082, 1084, 1086, 1087, 1090], metadata={}, asm_lines=[
            'iload  0x06',
            'iload  0x07',
            'iadd',
            'sipush 0x07d0<2000>',
            'if_icmpne 0x000d',
        ]),
        1093: CFGBasicBlock(parent_function=__auto_functions[945], address=1093, asm_memory_addresses=[1093, 1096, 1097], metadata={}, asm_lines=[
            'getstatic 0x001c<java/lang/System::out>',
            'iconst_0',
            'invokevirtual 0x0020<java/io/PrintStream::println>',
        ]),
        1100: CFGBasicBlock(parent_function=__auto_functions[945], address=1100, asm_memory_addresses=[1100], metadata={}, asm_lines=[
            'goto   0x0010',
        ]),
        1103: CFGBasicBlock(parent_function=__auto_functions[945], address=1103, asm_memory_addresses=[1103, 1106, 1108, 1110], metadata={}, asm_lines=[
            'getstatic 0x001c<java/lang/System::out>',
            'iload  0x06',
            'iload  0x07',
            'invokestatic 0x0026<java/lang/Math::min>',
        ]),
        1113: CFGBasicBlock(parent_function=__auto_functions[945], address=1113, asm_memory_addresses=[1113], metadata={}, asm_lines=[
            'invokevirtual 0x0020<java/io/PrintStream::println>',
        ]),
        1116: CFGBasicBlock(parent_function=__auto_functions[945], address=1116, asm_memory_addresses=[1116], metadata={}, asm_lines=[
            'return',
        ]),
        1295: CFGBasicBlock(parent_function=__auto_functions[1295], address=1295, asm_memory_addresses=[1295, 1296, 1297, 1298, 1299, 1300, 1301, 1302], metadata={}, asm_lines=[
            'aload_0',
            'arraylength',
            'istore_2',
            'iload_2',
            'iload_1',
            'iconst_1',
            'iadd',
            'if_icmpne 0x002c',
        ]),
        1305: CFGBasicBlock(parent_function=__auto_functions[1295], address=1305, asm_memory_addresses=[1305, 1306, 1308, 1309, 1310], metadata={}, asm_lines=[
            'iload_2',
            'newarray 0x0a',
            'astore_3',
            'iconst_0',
            'istore 0x04',
        ]),
        1312: CFGBasicBlock(parent_function=__auto_functions[1295], address=1312, asm_memory_addresses=[1312, 1314, 1315, 1316], metadata={}, asm_lines=[
            'iload  0x04',
            'aload_3',
            'arraylength',
            'if_icmpge 0x0011',
        ]),
        1319: CFGBasicBlock(parent_function=__auto_functions[1295], address=1319, asm_memory_addresses=[1319, 1320, 1322, 1323, 1325, 1326, 1327, 1330], metadata={}, asm_lines=[
            'aload_3',
            'iload  0x04',
            'aload_0',
            'iload  0x04',
            'iaload',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffee<-18>',
        ]),
        1333: CFGBasicBlock(parent_function=__auto_functions[1295], address=1333, asm_memory_addresses=[1333, 1336, 1337], metadata={}, asm_lines=[
            'getstatic 0x002a<CF::list>',
            'aload_3',
            'invokeinterface 0x0030<java/util/List::add>, 0x02, 0x00',
        ]),
        1342: CFGBasicBlock(parent_function=__auto_functions[1295], address=1342, asm_memory_addresses=[1342, 1343], metadata={}, asm_lines=[
            'pop',
            'goto   0x0037',
        ]),
        1346: CFGBasicBlock(parent_function=__auto_functions[1295], address=1346, asm_memory_addresses=[1346, 1347], metadata={}, asm_lines=[
            'iload_1',
            'istore_3',
        ]),
        1348: CFGBasicBlock(parent_function=__auto_functions[1295], address=1348, asm_memory_addresses=[1348, 1349, 1350], metadata={}, asm_lines=[
            'iload_3',
            'iload_2',
            'if_icmpge 0x0030',
        ]),
        1353: CFGBasicBlock(parent_function=__auto_functions[1295], address=1353, asm_memory_addresses=[1353, 1354, 1355, 1356, 1358, 1359, 1360, 1361, 1362, 1363, 1364, 1365, 1366, 1368, 1369, 1370, 1371, 1372, 1373], metadata={}, asm_lines=[
            'aload_0',
            'iload_3',
            'iaload',
            'istore 0x04',
            'aload_0',
            'iload_3',
            'aload_0',
            'iload_1',
            'iaload',
            'iastore',
            'aload_0',
            'iload_1',
            'iload  0x04',
            'iastore',
            'aload_0',
            'iload_1',
            'iconst_1',
            'iadd',
            'invokestatic 0x0036<CF::Permute>',
        ]),
        1376: CFGBasicBlock(parent_function=__auto_functions[1295], address=1376, asm_memory_addresses=[1376, 1377, 1378, 1379, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1391, 1392, 1395], metadata={}, asm_lines=[
            'aload_0',
            'iload_3',
            'iaload',
            'istore 0x05',
            'aload_0',
            'iload_3',
            'aload_0',
            'iload_1',
            'iaload',
            'iastore',
            'aload_0',
            'iload_1',
            'iload  0x05',
            'iastore',
            'iinc   0x03, 0x01',
            'goto   0xffd1<-47>',
        ]),
        1398: CFGBasicBlock(parent_function=__auto_functions[1295], address=1398, asm_memory_addresses=[1398], metadata={}, asm_lines=[
            'return',
        ]),
        1534: CFGBasicBlock(parent_function=__auto_functions[1534], address=1534, asm_memory_addresses=[1534, 1535, 1536, 1537], metadata={}, asm_lines=[
            'aload_0',
            'aload_0',
            'arraylength',
            'invokestatic 0x003a<CF::radixSort>',
        ]),
        1540: CFGBasicBlock(parent_function=__auto_functions[1534], address=1540, asm_memory_addresses=[1540], metadata={}, asm_lines=[
            'areturn',
        ]),
        1579: CFGBasicBlock(parent_function=__auto_functions[1579], address=1579, asm_memory_addresses=[1579, 1580, 1582, 1583, 1585, 1587, 1588, 1589], metadata={}, asm_lines=[
            'iload_1',
            'newarray 0x0a',
            'astore_2',
            'ldc    0x3e',
            'newarray 0x0a',
            'astore_3',
            'iconst_0',
            'istore 0x04',
        ]),
        1591: CFGBasicBlock(parent_function=__auto_functions[1579], address=1591, asm_memory_addresses=[1591, 1593, 1594], metadata={}, asm_lines=[
            'iload  0x04',
            'iload_1',
            'if_icmpge 0x0018',
        ]),
        1597: CFGBasicBlock(parent_function=__auto_functions[1579], address=1597, asm_memory_addresses=[1597, 1598, 1599, 1600, 1602, 1603, 1605, 1606, 1607, 1608, 1609, 1610, 1611, 1612, 1615], metadata={}, asm_lines=[
            'aload_3',
            'iconst_1',
            'aload_0',
            'iload  0x04',
            'iaload',
            'ldc    0x3f',
            'iand',
            'iadd',
            'dup2',
            'iaload',
            'iconst_1',
            'iadd',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffe8<-24>',
        ]),
        1618: CFGBasicBlock(parent_function=__auto_functions[1579], address=1618, asm_memory_addresses=[1618, 1619], metadata={}, asm_lines=[
            'iconst_1',
            'istore 0x04',
        ]),
        1621: CFGBasicBlock(parent_function=__auto_functions[1579], address=1621, asm_memory_addresses=[1621, 1623, 1625], metadata={}, asm_lines=[
            'iload  0x04',
            'ldc    0x40',
            'if_icmpgt 0x0016',
        ]),
        1628: CFGBasicBlock(parent_function=__auto_functions[1579], address=1628, asm_memory_addresses=[1628, 1629, 1631, 1632, 1633, 1634, 1636, 1637, 1638, 1639, 1640, 1641, 1644], metadata={}, asm_lines=[
            'aload_3',
            'iload  0x04',
            'dup2',
            'iaload',
            'aload_3',
            'iload  0x04',
            'iconst_1',
            'isub',
            'iaload',
            'iadd',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffe9<-23>',
        ]),
        1647: CFGBasicBlock(parent_function=__auto_functions[1579], address=1647, asm_memory_addresses=[1647, 1648], metadata={}, asm_lines=[
            'iconst_0',
            'istore 0x04',
        ]),
        1650: CFGBasicBlock(parent_function=__auto_functions[1579], address=1650, asm_memory_addresses=[1650, 1652, 1653], metadata={}, asm_lines=[
            'iload  0x04',
            'iload_1',
            'if_icmpge 0x001d',
        ]),
        1656: CFGBasicBlock(parent_function=__auto_functions[1579], address=1656, asm_memory_addresses=[1656, 1657, 1658, 1659, 1661, 1662, 1664, 1665, 1666, 1667, 1668, 1669, 1670, 1671, 1672, 1674, 1675, 1676, 1679], metadata={}, asm_lines=[
            'aload_2',
            'aload_3',
            'aload_0',
            'iload  0x04',
            'iaload',
            'ldc    0x3f',
            'iand',
            'dup2',
            'iaload',
            'dup_x2',
            'iconst_1',
            'iadd',
            'iastore',
            'aload_0',
            'iload  0x04',
            'iaload',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffe3<-29>',
        ]),
        1682: CFGBasicBlock(parent_function=__auto_functions[1579], address=1682, asm_memory_addresses=[1682, 1683, 1685, 1686, 1687, 1689, 1690, 1692, 1694, 1695, 1696], metadata={}, asm_lines=[
            'aload_0',
            'astore 0x04',
            'aload_2',
            'astore_0',
            'aload  0x04',
            'astore_2',
            'ldc    0x3e',
            'newarray 0x0a',
            'astore_3',
            'iconst_0',
            'istore 0x04',
        ]),
        1698: CFGBasicBlock(parent_function=__auto_functions[1579], address=1698, asm_memory_addresses=[1698, 1700, 1701], metadata={}, asm_lines=[
            'iload  0x04',
            'iload_1',
            'if_icmpge 0x0018',
        ]),
        1704: CFGBasicBlock(parent_function=__auto_functions[1579], address=1704, asm_memory_addresses=[1704, 1705, 1706, 1707, 1709, 1710, 1712, 1713, 1714, 1715, 1716, 1717, 1718, 1719, 1722], metadata={}, asm_lines=[
            'aload_3',
            'iconst_1',
            'aload_0',
            'iload  0x04',
            'iaload',
            'bipush 0x10',
            'iushr',
            'iadd',
            'dup2',
            'iaload',
            'iconst_1',
            'iadd',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffe8<-24>',
        ]),
        1725: CFGBasicBlock(parent_function=__auto_functions[1579], address=1725, asm_memory_addresses=[1725, 1726], metadata={}, asm_lines=[
            'iconst_1',
            'istore 0x04',
        ]),
        1728: CFGBasicBlock(parent_function=__auto_functions[1579], address=1728, asm_memory_addresses=[1728, 1730, 1732], metadata={}, asm_lines=[
            'iload  0x04',
            'ldc    0x40',
            'if_icmpgt 0x0016',
        ]),
        1735: CFGBasicBlock(parent_function=__auto_functions[1579], address=1735, asm_memory_addresses=[1735, 1736, 1738, 1739, 1740, 1741, 1743, 1744, 1745, 1746, 1747, 1748, 1751], metadata={}, asm_lines=[
            'aload_3',
            'iload  0x04',
            'dup2',
            'iaload',
            'aload_3',
            'iload  0x04',
            'iconst_1',
            'isub',
            'iaload',
            'iadd',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffe9<-23>',
        ]),
        1754: CFGBasicBlock(parent_function=__auto_functions[1579], address=1754, asm_memory_addresses=[1754, 1755], metadata={}, asm_lines=[
            'iconst_0',
            'istore 0x04',
        ]),
        1757: CFGBasicBlock(parent_function=__auto_functions[1579], address=1757, asm_memory_addresses=[1757, 1759, 1760], metadata={}, asm_lines=[
            'iload  0x04',
            'iload_1',
            'if_icmpge 0x001d',
        ]),
        1763: CFGBasicBlock(parent_function=__auto_functions[1579], address=1763, asm_memory_addresses=[1763, 1764, 1765, 1766, 1768, 1769, 1771, 1772, 1773, 1774, 1775, 1776, 1777, 1778, 1779, 1781, 1782, 1783, 1786], metadata={}, asm_lines=[
            'aload_2',
            'aload_3',
            'aload_0',
            'iload  0x04',
            'iaload',
            'bipush 0x10',
            'iushr',
            'dup2',
            'iaload',
            'dup_x2',
            'iconst_1',
            'iadd',
            'iastore',
            'aload_0',
            'iload  0x04',
            'iaload',
            'iastore',
            'iinc   0x04, 0x01',
            'goto   0xffe3<-29>',
        ]),
        1789: CFGBasicBlock(parent_function=__auto_functions[1579], address=1789, asm_memory_addresses=[1789, 1790, 1792, 1793, 1794, 1796, 1797, 1798], metadata={}, asm_lines=[
            'aload_0',
            'astore 0x04',
            'aload_2',
            'astore_0',
            'aload  0x04',
            'astore_2',
            'aload_0',
            'areturn',
        ]),
        2001: CFGBasicBlock(parent_function=__auto_functions[2001], address=2001, asm_memory_addresses=[2001, 2004, 2005], metadata={}, asm_lines=[
            'new    0x0041<java/util/LinkedList>',
            'dup',
            'invokespecial 0x0043<java/util/LinkedList::<init>>',
        ]),
        2008: CFGBasicBlock(parent_function=__auto_functions[2001], address=2008, asm_memory_addresses=[2008, 2011], metadata={}, asm_lines=[
            'putstatic 0x002a<CF::list>',
            'return',
        ]),
        5216: CFGBasicBlock(parent_function=__auto_functions[5216], address=5216, asm_memory_addresses=[5216, 5217], metadata={}, asm_lines=[
            'aload_0',
            'invokespecial 0x0001<java/lang/Object::<init>>',
        ]),
        5220: CFGBasicBlock(parent_function=__auto_functions[5216], address=5220, asm_memory_addresses=[5220, 5221, 5224, 5226, 5229, 5230, 5231, 5234], metadata={}, asm_lines=[
            'aload_0',
            'sipush 0x0400<1024>',
            'newarray 0x08',
            'putfield 0x0007',
            'aload_0',
            'aload_1',
            'putfield 0x000d',
            'return',
        ]),
        5285: CFGBasicBlock(parent_function=__auto_functions[5285], address=5285, asm_memory_addresses=[5285, 5286, 5289, 5290], metadata={}, asm_lines=[
            'aload_0',
            'getfield 0x0011',
            'iconst_m1',
            'if_icmpne 0x000b',
        ]),
        5293: CFGBasicBlock(parent_function=__auto_functions[5285], address=5293, asm_memory_addresses=[5293, 5296, 5297], metadata={}, asm_lines=[
            'new    0x0015<java/util/InputMismatchException>',
            'dup',
            'invokespecial 0x0017<java/util/InputMismatchException::<init>>',
        ]),
        5300: CFGBasicBlock(parent_function=__auto_functions[5285], address=5300, asm_memory_addresses=[5300], metadata={}, asm_lines=[
            'athrow',
        ]),
        5301: CFGBasicBlock(parent_function=__auto_functions[5285], address=5301, asm_memory_addresses=[5301, 5302, 5305, 5306, 5309], metadata={}, asm_lines=[
            'aload_0',
            'getfield 0x0018',
            'aload_0',
            'getfield 0x0011',
            'if_icmplt 0x002c',
        ]),
        5312: CFGBasicBlock(parent_function=__auto_functions[5285], address=5312, asm_memory_addresses=[5312, 5313, 5314, 5317, 5318, 5319, 5322, 5323, 5326], metadata={}, asm_lines=[
            'aload_0',
            'iconst_0',
            'putfield 0x0018',
            'aload_0',
            'aload_0',
            'getfield 0x000d',
            'aload_0',
            'getfield 0x0007',
            'invokevirtual 0x001b<java/io/InputStream::read>',
        ]),
        5329: CFGBasicBlock(parent_function=__auto_functions[5285], address=5329, asm_memory_addresses=[5329, 5332], metadata={}, asm_lines=[
            'putfield 0x0011',
            'goto   0x000c',
        ]),
        5335: CFGBasicBlock(parent_function=__auto_functions[5285], address=5335, asm_memory_addresses=[5335, 5336, 5339, 5340], metadata={}, asm_lines=[
            'astore_1',
            'new    0x0015<java/util/InputMismatchException>',
            'dup',
            'invokespecial 0x0017<java/util/InputMismatchException::<init>>',
        ]),
        5343: CFGBasicBlock(parent_function=__auto_functions[5285], address=5343, asm_memory_addresses=[5343], metadata={}, asm_lines=[
            'athrow',
        ]),
        5344: CFGBasicBlock(parent_function=__auto_functions[5285], address=5344, asm_memory_addresses=[5344, 5345, 5348], metadata={}, asm_lines=[
            'aload_0',
            'getfield 0x0011',
            'ifgt   0x0005',
        ]),
        5351: CFGBasicBlock(parent_function=__auto_functions[5285], address=5351, asm_memory_addresses=[5351, 5352], metadata={}, asm_lines=[
            'iconst_m1',
            'ireturn',
        ]),
        5353: CFGBasicBlock(parent_function=__auto_functions[5285], address=5353, asm_memory_addresses=[5353, 5354, 5357, 5358, 5359, 5362, 5363, 5364, 5365, 5368, 5369], metadata={}, asm_lines=[
            'aload_0',
            'getfield 0x0007',
            'aload_0',
            'dup',
            'getfield 0x0018',
            'dup_x1',
            'iconst_1',
            'iadd',
            'putfield 0x0018',
            'baload',
            'ireturn',
        ]),
        5471: CFGBasicBlock(parent_function=__auto_functions[5471], address=5471, asm_memory_addresses=[5471, 5472, 5474], metadata={}, asm_lines=[
            'iload_1',
            'bipush 0x20',
            'if_icmpeq 0x001a',
        ]),
        5477: CFGBasicBlock(parent_function=__auto_functions[5471], address=5477, asm_memory_addresses=[5477, 5478, 5480], metadata={}, asm_lines=[
            'iload_1',
            'bipush 0x0a',
            'if_icmpeq 0x0014',
        ]),
        5483: CFGBasicBlock(parent_function=__auto_functions[5471], address=5483, asm_memory_addresses=[5483, 5484, 5486], metadata={}, asm_lines=[
            'iload_1',
            'bipush 0x0d',
            'if_icmpeq 0x000e',
        ]),
        5489: CFGBasicBlock(parent_function=__auto_functions[5471], address=5489, asm_memory_addresses=[5489, 5490, 5492], metadata={}, asm_lines=[
            'iload_1',
            'bipush 0x09',
            'if_icmpeq 0x0008',
        ]),
        5495: CFGBasicBlock(parent_function=__auto_functions[5471], address=5495, asm_memory_addresses=[5495, 5496, 5497], metadata={}, asm_lines=[
            'iload_1',
            'iconst_m1',
            'if_icmpne 0x0007',
        ]),
        5500: CFGBasicBlock(parent_function=__auto_functions[5471], address=5500, asm_memory_addresses=[5500, 5501], metadata={}, asm_lines=[
            'iconst_1',
            'goto   0x0004',
        ]),
        5504: CFGBasicBlock(parent_function=__auto_functions[5471], address=5504, asm_memory_addresses=[5504], metadata={}, asm_lines=[
            'iconst_0',
        ]),
        5505: CFGBasicBlock(parent_function=__auto_functions[5471], address=5505, asm_memory_addresses=[5505], metadata={}, asm_lines=[
            'ireturn',
        ]),
        5556: CFGBasicBlock(parent_function=__auto_functions[5556], address=5556, asm_memory_addresses=[5556, 5557, 5559], metadata={}, asm_lines=[
            'iload_1',
            'bipush 0x0a',
            'if_icmpeq 0x000e',
        ]),
        5562: CFGBasicBlock(parent_function=__auto_functions[5556], address=5562, asm_memory_addresses=[5562, 5563, 5565], metadata={}, asm_lines=[
            'iload_1',
            'bipush 0x0d',
            'if_icmpeq 0x0008',
        ]),
        5568: CFGBasicBlock(parent_function=__auto_functions[5556], address=5568, asm_memory_addresses=[5568, 5569, 5570], metadata={}, asm_lines=[
            'iload_1',
            'iconst_m1',
            'if_icmpne 0x0007',
        ]),
        5573: CFGBasicBlock(parent_function=__auto_functions[5556], address=5573, asm_memory_addresses=[5573, 5574], metadata={}, asm_lines=[
            'iconst_1',
            'goto   0x0004',
        ]),
        5577: CFGBasicBlock(parent_function=__auto_functions[5556], address=5577, asm_memory_addresses=[5577], metadata={}, asm_lines=[
            'iconst_0',
        ]),
        5578: CFGBasicBlock(parent_function=__auto_functions[5556], address=5578, asm_memory_addresses=[5578], metadata={}, asm_lines=[
            'ireturn',
        ]),
        5629: CFGBasicBlock(parent_function=__auto_functions[5629], address=5629, asm_memory_addresses=[5629, 5630], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x0023<CF$FastScanner::next>',
        ]),
        5633: CFGBasicBlock(parent_function=__auto_functions[5629], address=5633, asm_memory_addresses=[5633], metadata={}, asm_lines=[
            'invokestatic 0x0027<java/lang/Integer::parseInt>',
        ]),
        5636: CFGBasicBlock(parent_function=__auto_functions[5629], address=5636, asm_memory_addresses=[5636], metadata={}, asm_lines=[
            'ireturn',
        ]),
        5675: CFGBasicBlock(parent_function=__auto_functions[5675], address=5675, asm_memory_addresses=[5675, 5676, 5678, 5679, 5680], metadata={}, asm_lines=[
            'iload_1',
            'newarray 0x0a',
            'astore_2',
            'iconst_0',
            'istore_3',
        ]),
        5681: CFGBasicBlock(parent_function=__auto_functions[5675], address=5681, asm_memory_addresses=[5681, 5682, 5683], metadata={}, asm_lines=[
            'iload_3',
            'iload_1',
            'if_icmpge 0x0010',
        ]),
        5686: CFGBasicBlock(parent_function=__auto_functions[5675], address=5686, asm_memory_addresses=[5686, 5687, 5688, 5689], metadata={}, asm_lines=[
            'aload_2',
            'iload_3',
            'aload_0',
            'invokevirtual 0x002d<CF$FastScanner::nextInt>',
        ]),
        5692: CFGBasicBlock(parent_function=__auto_functions[5675], address=5692, asm_memory_addresses=[5692, 5693, 5696], metadata={}, asm_lines=[
            'iastore',
            'iinc   0x03, 0x01',
            'goto   0xfff1<-15>',
        ]),
        5699: CFGBasicBlock(parent_function=__auto_functions[5675], address=5699, asm_memory_addresses=[5699, 5700], metadata={}, asm_lines=[
            'aload_2',
            'areturn',
        ]),
        5773: CFGBasicBlock(parent_function=__auto_functions[5773], address=5773, asm_memory_addresses=[5773, 5774, 5777, 5778, 5779], metadata={}, asm_lines=[
            'iload_1',
            'anewarray 0x0031',
            'astore_2',
            'iconst_0',
            'istore_3',
        ]),
        5780: CFGBasicBlock(parent_function=__auto_functions[5773], address=5780, asm_memory_addresses=[5780, 5781, 5782], metadata={}, asm_lines=[
            'iload_3',
            'iload_1',
            'if_icmpge 0x0010',
        ]),
        5785: CFGBasicBlock(parent_function=__auto_functions[5773], address=5785, asm_memory_addresses=[5785, 5786, 5787, 5788], metadata={}, asm_lines=[
            'aload_2',
            'iload_3',
            'aload_0',
            'invokevirtual 0x0023<CF$FastScanner::next>',
        ]),
        5791: CFGBasicBlock(parent_function=__auto_functions[5773], address=5791, asm_memory_addresses=[5791, 5792, 5795], metadata={}, asm_lines=[
            'aastore',
            'iinc   0x03, 0x01',
            'goto   0xfff1<-15>',
        ]),
        5798: CFGBasicBlock(parent_function=__auto_functions[5773], address=5798, asm_memory_addresses=[5798, 5799], metadata={}, asm_lines=[
            'aload_2',
            'areturn',
        ]),
        5872: CFGBasicBlock(parent_function=__auto_functions[5872], address=5872, asm_memory_addresses=[5872, 5873], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x0023<CF$FastScanner::next>',
        ]),
        5876: CFGBasicBlock(parent_function=__auto_functions[5872], address=5876, asm_memory_addresses=[5876], metadata={}, asm_lines=[
            'invokestatic 0x0033<java/lang/Long::parseLong>',
        ]),
        5879: CFGBasicBlock(parent_function=__auto_functions[5872], address=5879, asm_memory_addresses=[5879], metadata={}, asm_lines=[
            'lreturn',
        ]),
        5918: CFGBasicBlock(parent_function=__auto_functions[5918], address=5918, asm_memory_addresses=[5918, 5919], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x0023<CF$FastScanner::next>',
        ]),
        5922: CFGBasicBlock(parent_function=__auto_functions[5918], address=5922, asm_memory_addresses=[5922], metadata={}, asm_lines=[
            'invokestatic 0x0039<java/lang/Double::parseDouble>',
        ]),
        5925: CFGBasicBlock(parent_function=__auto_functions[5918], address=5925, asm_memory_addresses=[5925], metadata={}, asm_lines=[
            'dreturn',
        ]),
        5964: CFGBasicBlock(parent_function=__auto_functions[5964], address=5964, asm_memory_addresses=[5964, 5965], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x003f<CF$FastScanner::read>',
        ]),
        5968: CFGBasicBlock(parent_function=__auto_functions[5964], address=5968, asm_memory_addresses=[5968], metadata={}, asm_lines=[
            'istore_1',
        ]),
        5969: CFGBasicBlock(parent_function=__auto_functions[5964], address=5969, asm_memory_addresses=[5969, 5970, 5971], metadata={}, asm_lines=[
            'aload_0',
            'iload_1',
            'invokevirtual 0x0041<CF$FastScanner::isSpaceChar>',
        ]),
        5974: CFGBasicBlock(parent_function=__auto_functions[5964], address=5974, asm_memory_addresses=[5974], metadata={}, asm_lines=[
            'ifeq   0x000b',
        ]),
        5977: CFGBasicBlock(parent_function=__auto_functions[5964], address=5977, asm_memory_addresses=[5977, 5978], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x003f<CF$FastScanner::read>',
        ]),
        5981: CFGBasicBlock(parent_function=__auto_functions[5964], address=5981, asm_memory_addresses=[5981, 5982], metadata={}, asm_lines=[
            'istore_1',
            'goto   0xfff3<-13>',
        ]),
        5985: CFGBasicBlock(parent_function=__auto_functions[5964], address=5985, asm_memory_addresses=[5985, 5988, 5989], metadata={}, asm_lines=[
            'new    0x0045<java/lang/StringBuilder>',
            'dup',
            'invokespecial 0x0047<java/lang/StringBuilder::<init>>',
        ]),
        5992: CFGBasicBlock(parent_function=__auto_functions[5964], address=5992, asm_memory_addresses=[5992], metadata={}, asm_lines=[
            'astore_2',
        ]),
        5993: CFGBasicBlock(parent_function=__auto_functions[5964], address=5993, asm_memory_addresses=[5993, 5994, 5995], metadata={}, asm_lines=[
            'aload_2',
            'iload_1',
            'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>',
        ]),
        5998: CFGBasicBlock(parent_function=__auto_functions[5964], address=5998, asm_memory_addresses=[5998, 5999, 6000], metadata={}, asm_lines=[
            'pop',
            'aload_0',
            'invokevirtual 0x003f<CF$FastScanner::read>',
        ]),
        6003: CFGBasicBlock(parent_function=__auto_functions[5964], address=6003, asm_memory_addresses=[6003, 6004, 6005, 6006], metadata={}, asm_lines=[
            'istore_1',
            'aload_0',
            'iload_1',
            'invokevirtual 0x0041<CF$FastScanner::isSpaceChar>',
        ]),
        6009: CFGBasicBlock(parent_function=__auto_functions[5964], address=6009, asm_memory_addresses=[6009], metadata={}, asm_lines=[
            'ifeq   0xfff0<-16>',
        ]),
        6012: CFGBasicBlock(parent_function=__auto_functions[5964], address=6012, asm_memory_addresses=[6012, 6013], metadata={}, asm_lines=[
            'aload_2',
            'invokevirtual 0x004c<java/lang/StringBuilder::toString>',
        ]),
        6016: CFGBasicBlock(parent_function=__auto_functions[5964], address=6016, asm_memory_addresses=[6016], metadata={}, asm_lines=[
            'areturn',
        ]),
        6102: CFGBasicBlock(parent_function=__auto_functions[6102], address=6102, asm_memory_addresses=[6102, 6103], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x003f<CF$FastScanner::read>',
        ]),
        6106: CFGBasicBlock(parent_function=__auto_functions[6102], address=6106, asm_memory_addresses=[6106], metadata={}, asm_lines=[
            'istore_1',
        ]),
        6107: CFGBasicBlock(parent_function=__auto_functions[6102], address=6107, asm_memory_addresses=[6107, 6108, 6109], metadata={}, asm_lines=[
            'aload_0',
            'iload_1',
            'invokevirtual 0x004f<CF$FastScanner::isEndline>',
        ]),
        6112: CFGBasicBlock(parent_function=__auto_functions[6102], address=6112, asm_memory_addresses=[6112], metadata={}, asm_lines=[
            'ifeq   0x000b',
        ]),
        6115: CFGBasicBlock(parent_function=__auto_functions[6102], address=6115, asm_memory_addresses=[6115, 6116], metadata={}, asm_lines=[
            'aload_0',
            'invokevirtual 0x003f<CF$FastScanner::read>',
        ]),
        6119: CFGBasicBlock(parent_function=__auto_functions[6102], address=6119, asm_memory_addresses=[6119, 6120], metadata={}, asm_lines=[
            'istore_1',
            'goto   0xfff3<-13>',
        ]),
        6123: CFGBasicBlock(parent_function=__auto_functions[6102], address=6123, asm_memory_addresses=[6123, 6126, 6127], metadata={}, asm_lines=[
            'new    0x0045<java/lang/StringBuilder>',
            'dup',
            'invokespecial 0x0047<java/lang/StringBuilder::<init>>',
        ]),
        6130: CFGBasicBlock(parent_function=__auto_functions[6102], address=6130, asm_memory_addresses=[6130], metadata={}, asm_lines=[
            'astore_2',
        ]),
        6131: CFGBasicBlock(parent_function=__auto_functions[6102], address=6131, asm_memory_addresses=[6131, 6132, 6133], metadata={}, asm_lines=[
            'aload_2',
            'iload_1',
            'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>',
        ]),
        6136: CFGBasicBlock(parent_function=__auto_functions[6102], address=6136, asm_memory_addresses=[6136, 6137, 6138], metadata={}, asm_lines=[
            'pop',
            'aload_0',
            'invokevirtual 0x003f<CF$FastScanner::read>',
        ]),
        6141: CFGBasicBlock(parent_function=__auto_functions[6102], address=6141, asm_memory_addresses=[6141, 6142, 6143, 6144], metadata={}, asm_lines=[
            'istore_1',
            'aload_0',
            'iload_1',
            'invokevirtual 0x004f<CF$FastScanner::isEndline>',
        ]),
        6147: CFGBasicBlock(parent_function=__auto_functions[6102], address=6147, asm_memory_addresses=[6147], metadata={}, asm_lines=[
            'ifeq   0xfff0<-16>',
        ]),
        6150: CFGBasicBlock(parent_function=__auto_functions[6102], address=6150, asm_memory_addresses=[6150, 6151], metadata={}, asm_lines=[
            'aload_2',
            'invokevirtual 0x004c<java/lang/StringBuilder::toString>',
        ]),
        6154: CFGBasicBlock(parent_function=__auto_functions[6102], address=6154, asm_memory_addresses=[6154], metadata={}, asm_lines=[
            'areturn',
        ]),
        18446744073709508607: CFGBasicBlock(parent_function=__auto_functions[18446744073709508607], address=18446744073709508607, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709511679: CFGBasicBlock(parent_function=__auto_functions[18446744073709511679], address=18446744073709511679, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709512703: CFGBasicBlock(parent_function=__auto_functions[18446744073709512703], address=18446744073709512703, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709524991: CFGBasicBlock(parent_function=__auto_functions[18446744073709524991], address=18446744073709524991, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709527039: CFGBasicBlock(parent_function=__auto_functions[18446744073709527039], address=18446744073709527039, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709531135: CFGBasicBlock(parent_function=__auto_functions[18446744073709531135], address=18446744073709531135, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709533183: CFGBasicBlock(parent_function=__auto_functions[18446744073709533183], address=18446744073709533183, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709534207: CFGBasicBlock(parent_function=__auto_functions[18446744073709534207], address=18446744073709534207, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709536255: CFGBasicBlock(parent_function=__auto_functions[18446744073709536255], address=18446744073709536255, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709537279: CFGBasicBlock(parent_function=__auto_functions[18446744073709537279], address=18446744073709537279, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709540351: CFGBasicBlock(parent_function=__auto_functions[18446744073709540351], address=18446744073709540351, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709541375: CFGBasicBlock(parent_function=__auto_functions[18446744073709541375], address=18446744073709541375, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709542399: CFGBasicBlock(parent_function=__auto_functions[18446744073709542399], address=18446744073709542399, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
        18446744073709544447: CFGBasicBlock(parent_function=__auto_functions[18446744073709544447], address=18446744073709544447, asm_memory_addresses=[], metadata={}, asm_lines=[
            
        ]),
    }

    # Building all edges
    __auto_blocks[902].edges_out = set([
        CFGEdge(from_block=__auto_blocks[902], to_block=__auto_blocks[18446744073709536255], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[902], to_block=__auto_blocks[906], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[906].edges_out = set([
        
    ])

    __auto_blocks[945].edges_out = set([
        CFGEdge(from_block=__auto_blocks[945], to_block=__auto_blocks[5216], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[945], to_block=__auto_blocks[955], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[955].edges_out = set([
        CFGEdge(from_block=__auto_blocks[955], to_block=__auto_blocks[5629], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[955], to_block=__auto_blocks[960], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[960].edges_out = set([
        CFGEdge(from_block=__auto_blocks[960], to_block=__auto_blocks[5629], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[960], to_block=__auto_blocks[965], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[965].edges_out = set([
        CFGEdge(from_block=__auto_blocks[965], to_block=__auto_blocks[5629], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[965], to_block=__auto_blocks[970], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[970].edges_out = set([
        CFGEdge(from_block=__auto_blocks[970], to_block=__auto_blocks[5629], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[970], to_block=__auto_blocks[976], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[976].edges_out = set([
        CFGEdge(from_block=__auto_blocks[976], to_block=__auto_blocks[1024], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[976], to_block=__auto_blocks[987], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[987].edges_out = set([
        CFGEdge(from_block=__auto_blocks[987], to_block=__auto_blocks[18446744073709544447], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[987], to_block=__auto_blocks[996], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[996].edges_out = set([
        CFGEdge(from_block=__auto_blocks[996], to_block=__auto_blocks[1008], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[996], to_block=__auto_blocks[1030], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1008].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1008], to_block=__auto_blocks[1030], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1024].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1024], to_block=__auto_blocks[1030], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1030].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1030], to_block=__auto_blocks[1076], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1030], to_block=__auto_blocks[1039], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1039].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1039], to_block=__auto_blocks[18446744073709544447], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[1039], to_block=__auto_blocks[1048], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1048].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1048], to_block=__auto_blocks[1082], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1048], to_block=__auto_blocks[1060], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1060].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1060], to_block=__auto_blocks[1082], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1076].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1076], to_block=__auto_blocks[1082], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1082].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1082], to_block=__auto_blocks[1103], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1082], to_block=__auto_blocks[1093], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1093].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1093], to_block=__auto_blocks[18446744073709541375], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[1093], to_block=__auto_blocks[1100], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1100].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1100], to_block=__auto_blocks[1116], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1103].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1103], to_block=__auto_blocks[18446744073709542399], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[1103], to_block=__auto_blocks[1113], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1113].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1113], to_block=__auto_blocks[1116], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1113], to_block=__auto_blocks[18446744073709541375], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[1116].edges_out = set([
        
    ])

    __auto_blocks[1295].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1295], to_block=__auto_blocks[1305], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1295], to_block=__auto_blocks[1346], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1305].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1305], to_block=__auto_blocks[1312], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1312].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1312], to_block=__auto_blocks[1319], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1312], to_block=__auto_blocks[1333], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1319].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1319], to_block=__auto_blocks[1312], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1333].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1333], to_block=__auto_blocks[1342], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1333], to_block=__auto_blocks[18446744073709540351], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[1342].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1342], to_block=__auto_blocks[1398], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1346].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1346], to_block=__auto_blocks[1348], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1348].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1348], to_block=__auto_blocks[1398], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1348], to_block=__auto_blocks[1353], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1353].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1353], to_block=__auto_blocks[1295], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[1353], to_block=__auto_blocks[1376], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1376].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1376], to_block=__auto_blocks[1348], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1398].edges_out = set([
        
    ])

    __auto_blocks[1534].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1534], to_block=__auto_blocks[1540], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1534], to_block=__auto_blocks[1534], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[1540].edges_out = set([
        
    ])

    __auto_blocks[1579].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1579], to_block=__auto_blocks[1591], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1591].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1591], to_block=__auto_blocks[1618], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1591], to_block=__auto_blocks[1597], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1597].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1597], to_block=__auto_blocks[1591], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1618].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1618], to_block=__auto_blocks[1621], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1621].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1621], to_block=__auto_blocks[1647], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1621], to_block=__auto_blocks[1628], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1628].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1628], to_block=__auto_blocks[1621], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1647].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1647], to_block=__auto_blocks[1650], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1650].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1650], to_block=__auto_blocks[1656], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1650], to_block=__auto_blocks[1682], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1656].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1656], to_block=__auto_blocks[1650], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1682].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1682], to_block=__auto_blocks[1698], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1698].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1698], to_block=__auto_blocks[1704], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1698], to_block=__auto_blocks[1725], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1704].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1704], to_block=__auto_blocks[1698], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1725].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1725], to_block=__auto_blocks[1728], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1728].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1728], to_block=__auto_blocks[1754], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1728], to_block=__auto_blocks[1735], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1735].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1735], to_block=__auto_blocks[1728], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1754].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1754], to_block=__auto_blocks[1757], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1757].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1757], to_block=__auto_blocks[1789], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[1757], to_block=__auto_blocks[1763], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1763].edges_out = set([
        CFGEdge(from_block=__auto_blocks[1763], to_block=__auto_blocks[1757], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[1789].edges_out = set([
        
    ])

    __auto_blocks[2001].edges_out = set([
        CFGEdge(from_block=__auto_blocks[2001], to_block=__auto_blocks[2008], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[2001], to_block=__auto_blocks[18446744073709537279], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[2008].edges_out = set([
        
    ])

    __auto_blocks[5216].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5216], to_block=__auto_blocks[5220], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5216], to_block=__auto_blocks[18446744073709536255], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5220].edges_out = set([
        
    ])

    __auto_blocks[5285].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5285], to_block=__auto_blocks[5301], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5285], to_block=__auto_blocks[5293], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5293].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5293], to_block=__auto_blocks[18446744073709533183], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[5293], to_block=__auto_blocks[5300], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5300].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5300], to_block=__auto_blocks[5301], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5301].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5301], to_block=__auto_blocks[5312], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5301], to_block=__auto_blocks[5353], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5312].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5312], to_block=__auto_blocks[5329], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5312], to_block=__auto_blocks[18446744073709534207], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5329].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5329], to_block=__auto_blocks[5344], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5335].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5335], to_block=__auto_blocks[5343], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5335], to_block=__auto_blocks[18446744073709533183], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5343].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5343], to_block=__auto_blocks[5344], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5344].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5344], to_block=__auto_blocks[5351], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5344], to_block=__auto_blocks[5353], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5351].edges_out = set([
        
    ])

    __auto_blocks[5353].edges_out = set([
        
    ])

    __auto_blocks[5471].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5471], to_block=__auto_blocks[5500], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5471], to_block=__auto_blocks[5477], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5477].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5477], to_block=__auto_blocks[5500], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5477], to_block=__auto_blocks[5483], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5483].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5483], to_block=__auto_blocks[5489], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5483], to_block=__auto_blocks[5500], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5489].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5489], to_block=__auto_blocks[5500], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5489], to_block=__auto_blocks[5495], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5495].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5495], to_block=__auto_blocks[5504], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5495], to_block=__auto_blocks[5500], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5500].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5500], to_block=__auto_blocks[5505], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5504].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5504], to_block=__auto_blocks[5505], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5505].edges_out = set([
        
    ])

    __auto_blocks[5556].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5556], to_block=__auto_blocks[5562], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5556], to_block=__auto_blocks[5573], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5562].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5562], to_block=__auto_blocks[5573], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5562], to_block=__auto_blocks[5568], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5568].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5568], to_block=__auto_blocks[5573], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5568], to_block=__auto_blocks[5577], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5573].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5573], to_block=__auto_blocks[5578], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5577].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5577], to_block=__auto_blocks[5578], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5578].edges_out = set([
        
    ])

    __auto_blocks[5629].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5629], to_block=__auto_blocks[5633], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5633].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5633], to_block=__auto_blocks[18446744073709531135], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[5633], to_block=__auto_blocks[5636], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5636].edges_out = set([
        
    ])

    __auto_blocks[5675].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5675], to_block=__auto_blocks[5681], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5681].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5681], to_block=__auto_blocks[5699], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5681], to_block=__auto_blocks[5686], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5686].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5686], to_block=__auto_blocks[5692], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5686], to_block=__auto_blocks[5629], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5692].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5692], to_block=__auto_blocks[5681], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5699].edges_out = set([
        
    ])

    __auto_blocks[5773].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5773], to_block=__auto_blocks[5780], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5780].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5780], to_block=__auto_blocks[5785], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5780], to_block=__auto_blocks[5798], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5785].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5785], to_block=__auto_blocks[5791], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5791].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5791], to_block=__auto_blocks[5780], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5798].edges_out = set([
        
    ])

    __auto_blocks[5872].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5872], to_block=__auto_blocks[5876], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5876].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5876], to_block=__auto_blocks[5879], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5876], to_block=__auto_blocks[18446744073709527039], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5879].edges_out = set([
        
    ])

    __auto_blocks[5918].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5918], to_block=__auto_blocks[5922], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5922].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5922], to_block=__auto_blocks[18446744073709524991], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[5922], to_block=__auto_blocks[5925], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5925].edges_out = set([
        
    ])

    __auto_blocks[5964].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5964], to_block=__auto_blocks[5968], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5964], to_block=__auto_blocks[5285], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5968].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5968], to_block=__auto_blocks[5969], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5969].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5969], to_block=__auto_blocks[5974], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5969], to_block=__auto_blocks[5471], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5974].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5974], to_block=__auto_blocks[5985], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5974], to_block=__auto_blocks[5977], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5977].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5977], to_block=__auto_blocks[5285], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[5977], to_block=__auto_blocks[5981], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5981].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5981], to_block=__auto_blocks[5969], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5985].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5985], to_block=__auto_blocks[5992], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5985], to_block=__auto_blocks[18446744073709512703], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5992].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5992], to_block=__auto_blocks[5993], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[5993].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5993], to_block=__auto_blocks[5998], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[5993], to_block=__auto_blocks[18446744073709511679], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[5998].edges_out = set([
        CFGEdge(from_block=__auto_blocks[5998], to_block=__auto_blocks[5285], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[5998], to_block=__auto_blocks[6003], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6003].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6003], to_block=__auto_blocks[6009], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6003], to_block=__auto_blocks[5471], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[6009].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6009], to_block=__auto_blocks[6012], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6009], to_block=__auto_blocks[5993], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6012].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6012], to_block=__auto_blocks[6016], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6012], to_block=__auto_blocks[18446744073709508607], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[6016].edges_out = set([
        
    ])

    __auto_blocks[6102].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6102], to_block=__auto_blocks[5285], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[6102], to_block=__auto_blocks[6106], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6106].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6106], to_block=__auto_blocks[6107], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6107].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6107], to_block=__auto_blocks[6112], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6107], to_block=__auto_blocks[5556], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[6112].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6112], to_block=__auto_blocks[6115], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6112], to_block=__auto_blocks[6123], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6115].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6115], to_block=__auto_blocks[5285], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[6115], to_block=__auto_blocks[6119], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6119].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6119], to_block=__auto_blocks[6107], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6123].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6123], to_block=__auto_blocks[18446744073709512703], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[6123], to_block=__auto_blocks[6130], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6130].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6130], to_block=__auto_blocks[6131], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6131].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6131], to_block=__auto_blocks[6136], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6131], to_block=__auto_blocks[18446744073709511679], edge_type=EdgeType.FUNCTION_CALL),
    ])

    __auto_blocks[6136].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6136], to_block=__auto_blocks[5285], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[6136], to_block=__auto_blocks[6141], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6141].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6141], to_block=__auto_blocks[5556], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[6141], to_block=__auto_blocks[6147], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6147].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6147], to_block=__auto_blocks[6131], edge_type=EdgeType.NORMAL),
        CFGEdge(from_block=__auto_blocks[6147], to_block=__auto_blocks[6150], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6150].edges_out = set([
        CFGEdge(from_block=__auto_blocks[6150], to_block=__auto_blocks[18446744073709508607], edge_type=EdgeType.FUNCTION_CALL),
        CFGEdge(from_block=__auto_blocks[6150], to_block=__auto_blocks[6154], edge_type=EdgeType.NORMAL),
    ])

    __auto_blocks[6154].edges_out = set([
        
    ])

    __auto_blocks[18446744073709508607].edges_out = set([
        
    ])

    __auto_blocks[18446744073709511679].edges_out = set([
        
    ])

    __auto_blocks[18446744073709512703].edges_out = set([
        
    ])

    __auto_blocks[18446744073709524991].edges_out = set([
        
    ])

    __auto_blocks[18446744073709527039].edges_out = set([
        
    ])

    __auto_blocks[18446744073709531135].edges_out = set([
        
    ])

    __auto_blocks[18446744073709533183].edges_out = set([
        
    ])

    __auto_blocks[18446744073709534207].edges_out = set([
        
    ])

    __auto_blocks[18446744073709536255].edges_out = set([
        
    ])

    __auto_blocks[18446744073709537279].edges_out = set([
        
    ])

    __auto_blocks[18446744073709540351].edges_out = set([
        
    ])

    __auto_blocks[18446744073709541375].edges_out = set([
        
    ])

    __auto_blocks[18446744073709542399].edges_out = set([
        
    ])

    __auto_blocks[18446744073709544447].edges_out = set([
        
    ])


    # Set the edges_in on the blocks
    for b in __auto_blocks.values():
        for e in b.edges_out:
            e.to_block.edges_in.add(CFGEdge(b, e.to_block, e.edge_type))
            

    # Adding basic blocks to their associated functions
    __auto_functions[902].blocks = [
        __auto_blocks[902],
        __auto_blocks[906],
    ]

    __auto_functions[945].blocks = [
        __auto_blocks[945],
        __auto_blocks[955],
        __auto_blocks[960],
        __auto_blocks[965],
        __auto_blocks[970],
        __auto_blocks[976],
        __auto_blocks[987],
        __auto_blocks[1024],
        __auto_blocks[996],
        __auto_blocks[1008],
        __auto_blocks[1030],
        __auto_blocks[1039],
        __auto_blocks[1076],
        __auto_blocks[1048],
        __auto_blocks[1060],
        __auto_blocks[1082],
        __auto_blocks[1093],
        __auto_blocks[1103],
        __auto_blocks[1100],
        __auto_blocks[1116],
        __auto_blocks[1113],
    ]

    __auto_functions[1295].blocks = [
        __auto_blocks[1295],
        __auto_blocks[1305],
        __auto_blocks[1346],
        __auto_blocks[1312],
        __auto_blocks[1319],
        __auto_blocks[1333],
        __auto_blocks[1342],
        __auto_blocks[1398],
        __auto_blocks[1348],
        __auto_blocks[1353],
        __auto_blocks[1376],
    ]

    __auto_functions[1534].blocks = [
        __auto_blocks[1534],
        __auto_blocks[1540],
    ]

    __auto_functions[1579].blocks = [
        __auto_blocks[1579],
        __auto_blocks[1591],
        __auto_blocks[1597],
        __auto_blocks[1618],
        __auto_blocks[1621],
        __auto_blocks[1628],
        __auto_blocks[1647],
        __auto_blocks[1650],
        __auto_blocks[1656],
        __auto_blocks[1682],
        __auto_blocks[1698],
        __auto_blocks[1704],
        __auto_blocks[1725],
        __auto_blocks[1728],
        __auto_blocks[1735],
        __auto_blocks[1754],
        __auto_blocks[1757],
        __auto_blocks[1763],
        __auto_blocks[1789],
    ]

    __auto_functions[2001].blocks = [
        __auto_blocks[2001],
        __auto_blocks[2008],
    ]

    __auto_functions[5216].blocks = [
        __auto_blocks[5216],
        __auto_blocks[5220],
    ]

    __auto_functions[5285].blocks = [
        __auto_blocks[5285],
        __auto_blocks[5293],
        __auto_blocks[5301],
        __auto_blocks[5300],
        __auto_blocks[5312],
        __auto_blocks[5353],
        __auto_blocks[5329],
        __auto_blocks[5344],
        __auto_blocks[5335],
        __auto_blocks[5343],
        __auto_blocks[5351],
    ]

    __auto_functions[5471].blocks = [
        __auto_blocks[5471],
        __auto_blocks[5477],
        __auto_blocks[5500],
        __auto_blocks[5483],
        __auto_blocks[5489],
        __auto_blocks[5495],
        __auto_blocks[5504],
        __auto_blocks[5505],
    ]

    __auto_functions[5556].blocks = [
        __auto_blocks[5556],
        __auto_blocks[5562],
        __auto_blocks[5573],
        __auto_blocks[5568],
        __auto_blocks[5577],
        __auto_blocks[5578],
    ]

    __auto_functions[5629].blocks = [
        __auto_blocks[5629],
        __auto_blocks[5633],
        __auto_blocks[5636],
    ]

    __auto_functions[5675].blocks = [
        __auto_blocks[5675],
        __auto_blocks[5681],
        __auto_blocks[5686],
        __auto_blocks[5699],
        __auto_blocks[5692],
    ]

    __auto_functions[5773].blocks = [
        __auto_blocks[5773],
        __auto_blocks[5780],
        __auto_blocks[5785],
        __auto_blocks[5798],
        __auto_blocks[5791],
    ]

    __auto_functions[5872].blocks = [
        __auto_blocks[5872],
        __auto_blocks[5876],
        __auto_blocks[5879],
    ]

    __auto_functions[5918].blocks = [
        __auto_blocks[5918],
        __auto_blocks[5922],
        __auto_blocks[5925],
    ]

    __auto_functions[5964].blocks = [
        __auto_blocks[5964],
        __auto_blocks[5968],
        __auto_blocks[5969],
        __auto_blocks[5974],
        __auto_blocks[5977],
        __auto_blocks[5985],
        __auto_blocks[5981],
        __auto_blocks[5992],
        __auto_blocks[5993],
        __auto_blocks[5998],
        __auto_blocks[6003],
        __auto_blocks[6009],
        __auto_blocks[6012],
        __auto_blocks[6016],
    ]

    __auto_functions[6102].blocks = [
        __auto_blocks[6102],
        __auto_blocks[6106],
        __auto_blocks[6107],
        __auto_blocks[6112],
        __auto_blocks[6115],
        __auto_blocks[6123],
        __auto_blocks[6119],
        __auto_blocks[6130],
        __auto_blocks[6131],
        __auto_blocks[6136],
        __auto_blocks[6141],
        __auto_blocks[6147],
        __auto_blocks[6150],
        __auto_blocks[6154],
    ]

    __auto_functions[18446744073709508607].blocks = [
        __auto_blocks[18446744073709508607],
    ]

    __auto_functions[18446744073709511679].blocks = [
        __auto_blocks[18446744073709511679],
    ]

    __auto_functions[18446744073709512703].blocks = [
        __auto_blocks[18446744073709512703],
    ]

    __auto_functions[18446744073709524991].blocks = [
        __auto_blocks[18446744073709524991],
    ]

    __auto_functions[18446744073709527039].blocks = [
        __auto_blocks[18446744073709527039],
    ]

    __auto_functions[18446744073709531135].blocks = [
        __auto_blocks[18446744073709531135],
    ]

    __auto_functions[18446744073709533183].blocks = [
        __auto_blocks[18446744073709533183],
    ]

    __auto_functions[18446744073709534207].blocks = [
        __auto_blocks[18446744073709534207],
    ]

    __auto_functions[18446744073709536255].blocks = [
        __auto_blocks[18446744073709536255],
    ]

    __auto_functions[18446744073709537279].blocks = [
        __auto_blocks[18446744073709537279],
    ]

    __auto_functions[18446744073709540351].blocks = [
        __auto_blocks[18446744073709540351],
    ]

    __auto_functions[18446744073709541375].blocks = [
        __auto_blocks[18446744073709541375],
    ]

    __auto_functions[18446744073709542399].blocks = [
        __auto_blocks[18446744073709542399],
    ]

    __auto_functions[18446744073709544447].blocks = [
        __auto_blocks[18446744073709544447],
    ]

    
    expected = {
        'sorted_func_order': [902, 945, 1295, 1534, 1579, 2001, 5216, 5285, 5471, 5556, 5629, 5675, 5773, 5872, 5918, 5964, 6102, 18446744073709508607, 18446744073709511679, 18446744073709512703, 18446744073709524991, 18446744073709527039, 18446744073709531135, 18446744073709533183, 18446744073709534207, 18446744073709536255, 18446744073709537279, 18446744073709540351, 18446744073709541375, 18446744073709542399, 18446744073709544447],
        'sorted_block_order': [902, 906, 945, 955, 960, 965, 970, 976, 987, 996, 1008, 1024, 1030, 1039, 1048, 1060, 1076, 1082, 1093, 1100, 1103, 1113, 1116, 1295, 1305, 1312, 1319, 1333, 1342, 1346, 1348, 1353, 1376, 1398, 1534, 1540, 1579, 1591, 1597, 1618, 1621, 1628, 1647, 1650, 1656, 1682, 1698, 1704, 1725, 1728, 1735, 1754, 1757, 1763, 1789, 2001, 2008, 5216, 5220, 5285, 5293, 5300, 5301, 5312, 5329, 5335, 5343, 5344, 5351, 5353, 5471, 5477, 5483, 5489, 5495, 5500, 5504, 5505, 5556, 5562, 5568, 5573, 5577, 5578, 5629, 5633, 5636, 5675, 5681, 5686, 5692, 5699, 5773, 5780, 5785, 5791, 5798, 5872, 5876, 5879, 5918, 5922, 5925, 5964, 5968, 5969, 5974, 5977, 5981, 5985, 5992, 5993, 5998, 6003, 6009, 6012, 6016, 6102, 6106, 6107, 6112, 6115, 6119, 6123, 6130, 6131, 6136, 6141, 6147, 6150, 6154, 18446744073709508607, 18446744073709511679, 18446744073709512703, 18446744073709524991, 18446744073709527039, 18446744073709531135, 18446744073709533183, 18446744073709534207, 18446744073709536255, 18446744073709537279, 18446744073709540351, 18446744073709541375, 18446744073709542399, 18446744073709544447],
        'architecture': 'java',
        'num_blocks': {902: 2, 945: 21, 1295: 11, 1534: 2, 1579: 19, 2001: 2, 5216: 2, 5285: 11, 5471: 8, 5556: 6, 5629: 3, 5675: 5, 5773: 5, 5872: 3, 5918: 3, 5964: 14, 6102: 14, 18446744073709508607: 1, 18446744073709511679: 1, 18446744073709512703: 1, 18446744073709524991: 1, 18446744073709527039: 1, 18446744073709531135: 1, 18446744073709533183: 1, 18446744073709534207: 1, 18446744073709536255: 1, 18446744073709537279: 1, 18446744073709540351: 1, 18446744073709541375: 1, 18446744073709542399: 1, 18446744073709544447: 1},
        'num_asm_lines_per_block': {902: 2, 906: 1, 945: 4, 955: 3, 960: 3, 965: 3, 970: 3, 976: 6, 987: 5, 996: 6, 1008: 8, 1024: 1, 1030: 5, 1039: 5, 1048: 6, 1060: 8, 1076: 1, 1082: 5, 1093: 3, 1100: 1, 1103: 4, 1113: 1, 1116: 1, 1295: 8, 1305: 5, 1312: 4, 1319: 8, 1333: 3, 1342: 2, 1346: 2, 1348: 3, 1353: 19, 1376: 16, 1398: 1, 1534: 4, 1540: 1, 1579: 8, 1591: 3, 1597: 15, 1618: 2, 1621: 3, 1628: 13, 1647: 2, 1650: 3, 1656: 19, 1682: 11, 1698: 3, 1704: 15, 1725: 2, 1728: 3, 1735: 13, 1754: 2, 1757: 3, 1763: 19, 1789: 8, 2001: 3, 2008: 2, 5216: 2, 5220: 8, 5285: 4, 5293: 3, 5300: 1, 5301: 5, 5312: 9, 5329: 2, 5335: 4, 5343: 1, 5344: 3, 5351: 2, 5353: 11, 5471: 3, 5477: 3, 5483: 3, 5489: 3, 5495: 3, 5500: 2, 5504: 1, 5505: 1, 5556: 3, 5562: 3, 5568: 3, 5573: 2, 5577: 1, 5578: 1, 5629: 2, 5633: 1, 5636: 1, 5675: 5, 5681: 3, 5686: 4, 5692: 3, 5699: 2, 5773: 5, 5780: 3, 5785: 4, 5791: 3, 5798: 2, 5872: 2, 5876: 1, 5879: 1, 5918: 2, 5922: 1, 5925: 1, 5964: 2, 5968: 1, 5969: 3, 5974: 1, 5977: 2, 5981: 2, 5985: 3, 5992: 1, 5993: 3, 5998: 3, 6003: 4, 6009: 1, 6012: 2, 6016: 1, 6102: 2, 6106: 1, 6107: 3, 6112: 1, 6115: 2, 6119: 2, 6123: 3, 6130: 1, 6131: 3, 6136: 3, 6141: 4, 6147: 1, 6150: 2, 6154: 1, 18446744073709508607: 0, 18446744073709511679: 0, 18446744073709512703: 0, 18446744073709524991: 0, 18446744073709527039: 0, 18446744073709531135: 0, 18446744073709533183: 0, 18446744073709534207: 0, 18446744073709536255: 0, 18446744073709537279: 0, 18446744073709540351: 0, 18446744073709541375: 0, 18446744073709542399: 0, 18446744073709544447: 0},
        'num_asm_lines_per_function': {902: 3, 945: 82, 1295: 71, 1534: 5, 1579: 147, 2001: 5, 5216: 10, 5285: 45, 5471: 19, 5556: 13, 5629: 4, 5675: 17, 5773: 17, 5872: 4, 5918: 4, 5964: 29, 6102: 29, 18446744073709508607: 0, 18446744073709511679: 0, 18446744073709512703: 0, 18446744073709524991: 0, 18446744073709527039: 0, 18446744073709531135: 0, 18446744073709533183: 0, 18446744073709534207: 0, 18446744073709536255: 0, 18446744073709537279: 0, 18446744073709540351: 0, 18446744073709541375: 0, 18446744073709542399: 0, 18446744073709544447: 0},
        'num_functions': 31,
        'is_root_function': {902: True, 945: True, 1295: False, 1534: False, 1579: True, 2001: True, 5216: False, 5285: False, 5471: False, 5556: False, 5629: False, 5675: True, 5773: True, 5872: True, 5918: True, 5964: True, 6102: True, 18446744073709508607: False, 18446744073709511679: False, 18446744073709512703: False, 18446744073709524991: False, 18446744073709527039: False, 18446744073709531135: False, 18446744073709533183: False, 18446744073709534207: False, 18446744073709536255: False, 18446744073709537279: False, 18446744073709540351: False, 18446744073709541375: False, 18446744073709542399: False, 18446744073709544447: False},
        'is_recursive': {902: False, 945: False, 1295: True, 1534: True, 1579: False, 2001: False, 5216: False, 5285: False, 5471: False, 5556: False, 5629: False, 5675: False, 5773: False, 5872: False, 5918: False, 5964: False, 6102: False, 18446744073709508607: False, 18446744073709511679: False, 18446744073709512703: False, 18446744073709524991: False, 18446744073709527039: False, 18446744073709531135: False, 18446744073709533183: False, 18446744073709534207: False, 18446744073709536255: False, 18446744073709537279: False, 18446744073709540351: False, 18446744073709541375: False, 18446744073709542399: False, 18446744073709544447: False},
        'is_extern_function': {902: False, 945: False, 1295: False, 1534: False, 1579: False, 2001: False, 5216: False, 5285: False, 5471: False, 5556: False, 5629: False, 5675: False, 5773: False, 5872: False, 5918: False, 5964: False, 6102: False, 18446744073709508607: False, 18446744073709511679: False, 18446744073709512703: False, 18446744073709524991: False, 18446744073709527039: False, 18446744073709531135: False, 18446744073709533183: False, 18446744073709534207: False, 18446744073709536255: False, 18446744073709537279: False, 18446744073709540351: False, 18446744073709541375: False, 18446744073709542399: False, 18446744073709544447: False},
        'is_intern_function': {902: True, 945: True, 1295: True, 1534: True, 1579: True, 2001: True, 5216: True, 5285: True, 5471: True, 5556: True, 5629: True, 5675: True, 5773: True, 5872: True, 5918: True, 5964: True, 6102: True, 18446744073709508607: True, 18446744073709511679: True, 18446744073709512703: True, 18446744073709524991: True, 18446744073709527039: True, 18446744073709531135: True, 18446744073709533183: True, 18446744073709534207: True, 18446744073709536255: True, 18446744073709537279: True, 18446744073709540351: True, 18446744073709541375: True, 18446744073709542399: True, 18446744073709544447: True},
        'function_entry_block': {902: 902, 945: 945, 1295: 1295, 1534: 1534, 1579: 1579, 2001: 2001, 5216: 5216, 5285: 5285, 5471: 5471, 5556: 5556, 5629: 5629, 5675: 5675, 5773: 5773, 5872: 5872, 5918: 5918, 5964: 5964, 6102: 6102, 18446744073709508607: 18446744073709508607, 18446744073709511679: 18446744073709511679, 18446744073709512703: 18446744073709512703, 18446744073709524991: 18446744073709524991, 18446744073709527039: 18446744073709527039, 18446744073709531135: 18446744073709531135, 18446744073709533183: 18446744073709533183, 18446744073709534207: 18446744073709534207, 18446744073709536255: 18446744073709536255, 18446744073709537279: 18446744073709537279, 18446744073709540351: 18446744073709540351, 18446744073709541375: 18446744073709541375, 18446744073709542399: 18446744073709542399, 18446744073709544447: 18446744073709544447},
        'called_by': {902: set(), 945: set(), 1295: {1353}, 1534: {1534}, 1579: set(), 2001: set(), 5216: {945}, 5285: {6115, 5964, 5998, 6102, 6136, 5977}, 5471: {5969, 6003}, 5556: {6107, 6141}, 5629: {960, 965, 970, 5686, 955}, 5675: set(), 5773: set(), 5872: set(), 5918: set(), 5964: set(), 6102: set(), 18446744073709508607: {6012, 6150}, 18446744073709511679: {5993, 6131}, 18446744073709512703: {5985, 6123}, 18446744073709524991: {5922}, 18446744073709527039: {5876}, 18446744073709531135: {5633}, 18446744073709533183: {5293, 5335}, 18446744073709534207: {5312}, 18446744073709536255: {5216, 902}, 18446744073709537279: {2001}, 18446744073709540351: {1333}, 18446744073709541375: {1113, 1093}, 18446744073709542399: {1103}, 18446744073709544447: {987, 1039}},
        'function_hashes': {902: 1565831104739572131, 945: 942950037572490334, 1295: 334174057499898723, 1534: 471672160074014773, 1579: 1089348810148860299, 2001: 265368743310362288, 5216: 611828272018515008, 5285: 2304109092604384749, 5471: 1177672058750724279, 5556: 1411753127854173831, 5629: 2181287607772695543, 5675: 1564382681207070681, 5773: 1396888527843700521, 5872: 1407806904075788791, 5918: 851130412926958137, 5964: 1249094458789108177, 6102: 810240923485578822, 18446744073709508607: 491466667472558797, 18446744073709511679: 1564439972551981801, 18446744073709512703: 360740132670044258, 18446744073709524991: 127740920699931385, 18446744073709527039: 2090814951510025067, 18446744073709531135: 2139520521362955294, 18446744073709533183: 372890675185618207, 18446744073709534207: 1721853280939839364, 18446744073709536255: 441802918820116996, 18446744073709537279: 1492900508615394748, 18446744073709540351: 689923618058421776, 18446744073709541375: 1862317289414483014, 18446744073709542399: 1281619765413699358, 18446744073709544447: 911852321925002949},
        'block_hashes': {902: 1481236198456797889, 906: 2009458719837634007, 945: 882749067433952708, 955: 796595485617035931, 960: 1605060496863044620, 965: 1832294820951533375, 970: 1150174361968655824, 976: 1564337719315585738, 987: 1871987102952149845, 996: 1523230693987109106, 1008: 762686789046566663, 1024: 2142430475542912764, 1030: 1743619817643079680, 1039: 2236489178600803326, 1048: 792175857812509416, 1060: 22843160817991409, 1076: 897318946611146212, 1082: 800737614490203370, 1093: 437943467124792746, 1100: 1478546889203870346, 1103: 1848290195697266961, 1113: 1558656136022127565, 1116: 635380143824509600, 1295: 1690199951362437394, 1305: 1959780398622337003, 1312: 1732739611986126039, 1319: 1605813590538238486, 1333: 167929666084463851, 1342: 756757190856465621, 1346: 480450221747033670, 1348: 745720259870860468, 1353: 531769583099324803, 1376: 500993334894879214, 1398: 1691272765203245851, 1534: 1934040203006570180, 1540: 230402586340516814, 1579: 1923141071830622005, 1591: 1138484938173036955, 1597: 313931698856314342, 1618: 1252451996278341214, 1621: 173938324078114540, 1628: 549718935812125828, 1647: 789614049268677317, 1650: 1294451291214945414, 1656: 559529951331491339, 1682: 1140983638649158858, 1698: 2075099350627172944, 1704: 1559995591579157129, 1725: 759826902561684304, 1728: 773810530558299801, 1735: 310131819345547272, 1754: 112283241524589782, 1757: 1190651110660260946, 1763: 401448646573709721, 1789: 1814328126257262015, 2001: 1438079933165796791, 2008: 1470680477817384839, 5216: 408571451000102988, 5220: 1302379645664622262, 5285: 313395355805132752, 5293: 1578390622938608525, 5300: 2247423818568232290, 5301: 1707455701044198339, 5312: 129334142456978343, 5329: 1993369748193069231, 5335: 161909507097350947, 5343: 1804615142700098353, 5344: 2004531038096590517, 5351: 1254818524555730001, 5353: 1099355224706335185, 5471: 2242114245624060450, 5477: 2101240373494876407, 5483: 1938112812446756386, 5489: 671111176047769761, 5495: 913209255682275111, 5500: 1865092588471318593, 5504: 1009995036482959686, 5505: 1364644787080988512, 5556: 2208032684620226405, 5562: 778382340685803525, 5568: 955219173666696203, 5573: 1354761108888546122, 5577: 1581670602097918420, 5578: 357703650988545601, 5629: 1032612367823506596, 5633: 2109961275466176390, 5636: 698109712948019164, 5675: 1114463919940810490, 5681: 212157950375588591, 5686: 360329123390409746, 5692: 2181324572257103787, 5699: 1630668910170987339, 5773: 2008438815913907395, 5780: 1526844599882168403, 5785: 1907665476128302070, 5791: 1902898958211288362, 5798: 117806363782782778, 5872: 2121107824102887628, 5876: 905102644426894457, 5879: 1170005587449961586, 5918: 1666094395021160111, 5922: 2142649522688022943, 5925: 1250209621642056675, 5964: 1152340493113044790, 5968: 1784084657711682609, 5969: 177781892342647796, 5974: 655439306899960012, 5977: 286977460121407108, 5981: 678966707190854857, 5985: 1395543445103151246, 5992: 1806072860810493552, 5993: 638186700345590931, 5998: 1545153187477998871, 6003: 1987252517410330260, 6009: 1535030253811040042, 6012: 1691066595067373189, 6016: 309096131091950739, 6102: 2040300218017475917, 6106: 188540807759714439, 6107: 2151878206086630661, 6112: 1302107828477164546, 6115: 848294301034910111, 6119: 1785063483569499434, 6123: 343964536648799383, 6130: 1593838252433939169, 6131: 1265239561007329719, 6136: 1359760804243143890, 6141: 345097191366304994, 6147: 473092642787469230, 6150: 427528838559718992, 6154: 161526992612518685, 18446744073709508607: 2082728530080268670, 18446744073709511679: 357207146564319631, 18446744073709512703: 1213942645591216453, 18446744073709524991: 577070441007554521, 18446744073709527039: 207540561290072648, 18446744073709531135: 1783429617477386212, 18446744073709533183: 492725414473427148, 18446744073709534207: 100892061406498973, 18446744073709536255: 2203474088995956417, 18446744073709537279: 338701221469707988, 18446744073709540351: 1277822881454820676, 18446744073709541375: 1743794292153176297, 18446744073709542399: 862805981993157638, 18446744073709544447: 274494280577352230},
        'cfg_hash': 567811153397826448,
        'memcfg_hashes': {'java_base-op': 507404490328818982, 'java_base-inst': 635685732558497970, 'java_repl_imm-op': 1883755188978701630, 'java_repl_imm-inst': 1627618868612256928},
        'metadata': {},
        'block_metadatas': {902: {}, 906: {}, 945: {}, 955: {}, 960: {}, 965: {}, 970: {}, 976: {}, 987: {}, 996: {}, 1008: {}, 1024: {}, 1030: {}, 1039: {}, 1048: {}, 1060: {}, 1076: {}, 1082: {}, 1093: {}, 1100: {}, 1103: {}, 1113: {}, 1116: {}, 1295: {}, 1305: {}, 1312: {}, 1319: {}, 1333: {}, 1342: {}, 1346: {}, 1348: {}, 1353: {}, 1376: {}, 1398: {}, 1534: {}, 1540: {}, 1579: {}, 1591: {}, 1597: {}, 1618: {}, 1621: {}, 1628: {}, 1647: {}, 1650: {}, 1656: {}, 1682: {}, 1698: {}, 1704: {}, 1725: {}, 1728: {}, 1735: {}, 1754: {}, 1757: {}, 1763: {}, 1789: {}, 2001: {}, 2008: {}, 5216: {}, 5220: {}, 5285: {}, 5293: {}, 5300: {}, 5301: {}, 5312: {}, 5329: {}, 5335: {}, 5343: {}, 5344: {}, 5351: {}, 5353: {}, 5471: {}, 5477: {}, 5483: {}, 5489: {}, 5495: {}, 5500: {}, 5504: {}, 5505: {}, 5556: {}, 5562: {}, 5568: {}, 5573: {}, 5577: {}, 5578: {}, 5629: {}, 5633: {}, 5636: {}, 5675: {}, 5681: {}, 5686: {}, 5692: {}, 5699: {}, 5773: {}, 5780: {}, 5785: {}, 5791: {}, 5798: {}, 5872: {}, 5876: {}, 5879: {}, 5918: {}, 5922: {}, 5925: {}, 5964: {}, 5968: {}, 5969: {}, 5974: {}, 5977: {}, 5981: {}, 5985: {}, 5992: {}, 5993: {}, 5998: {}, 6003: {}, 6009: {}, 6012: {}, 6016: {}, 6102: {}, 6106: {}, 6107: {}, 6112: {}, 6115: {}, 6119: {}, 6123: {}, 6130: {}, 6131: {}, 6136: {}, 6141: {}, 6147: {}, 6150: {}, 6154: {}, 18446744073709508607: {}, 18446744073709511679: {}, 18446744073709512703: {}, 18446744073709524991: {}, 18446744073709527039: {}, 18446744073709531135: {}, 18446744073709533183: {}, 18446744073709534207: {}, 18446744073709536255: {}, 18446744073709537279: {}, 18446744073709540351: {}, 18446744073709541375: {}, 18446744073709542399: {}, 18446744073709544447: {}},
        'function_metadatas': {902: {}, 945: {}, 1295: {}, 1534: {}, 1579: {}, 2001: {}, 5216: {}, 5285: {}, 5471: {}, 5556: {}, 5629: {}, 5675: {}, 5773: {}, 5872: {}, 5918: {}, 5964: {}, 6102: {}, 18446744073709508607: {}, 18446744073709511679: {}, 18446744073709512703: {}, 18446744073709524991: {}, 18446744073709527039: {}, 18446744073709531135: {}, 18446744073709533183: {}, 18446744073709534207: {}, 18446744073709536255: {}, 18446744073709537279: {}, 18446744073709540351: {}, 18446744073709541375: {}, 18446744073709542399: {}, 18446744073709544447: {}},
        'asm_counts_per_block': {
            902: {'aload_0': 1, 'invokespecial 0x0001<java/lang/Object::<init>>': 1},
            906: {'return': 1},
            945: {'new    0x0007<CF$FastScanner>': 1, 'dup': 1, 'getstatic 0x0009<java/lang/System::in>': 1, 'invokespecial 0x000f<CF$FastScanner::<init>>': 1},
            955: {'astore_1': 1, 'aload_1': 1, 'invokevirtual 0x0012<CF$FastScanner::nextInt>': 1},
            960: {'istore_2': 1, 'aload_1': 1, 'invokevirtual 0x0012<CF$FastScanner::nextInt>': 1},
            965: {'istore_3': 1, 'aload_1': 1, 'invokevirtual 0x0012<CF$FastScanner::nextInt>': 1},
            970: {'istore 0x04': 1, 'aload_1': 1, 'invokevirtual 0x0012<CF$FastScanner::nextInt>': 1},
            976: {'istore 0x05': 1, 'iconst_0': 1, 'istore 0x06': 1, 'iload  0x04': 1, 'iconst_1': 1, 'if_icmple 0x0028': 1},
            987: {'iload  0x06': 1, 'iload_3': 1, 'iload  0x04': 1, 'isub': 1, 'invokestatic 0x0016<java/lang/Math::abs>': 1},
            996: {'iadd': 1, 'istore 0x06': 1, 'iinc   0x06, 0x01': 1, 'iload  0x05': 1, 'iload_2': 1, 'if_icmpge 0x0019': 1},
            1008: {'iload  0x06': 1, 'iload  0x05': 1, 'iload  0x04': 1, 'isub': 1, 'iadd': 1, 'istore 0x06': 1, 'iinc   0x06, 0x01': 1, 'goto   0x0009': 1},
            1024: {'wide   0x84<-124>, 0x0006, 0x03e8<1000>': 1},
            1030: {'iconst_0': 1, 'istore 0x07': 1, 'iload  0x05': 1, 'iload_2': 1, 'if_icmpge 0x0028': 1},
            1039: {'iload  0x07': 1, 'iload_3': 1, 'iload  0x05': 1, 'isub': 1, 'invokestatic 0x0016<java/lang/Math::abs>': 1},
            1048: {'iadd': 1, 'istore 0x07': 1, 'iinc   0x07, 0x01': 1, 'iload  0x04': 1, 'iconst_1': 1, 'if_icmple 0x0019': 1},
            1060: {'iload  0x07': 1, 'iload  0x05': 1, 'iload  0x04': 1, 'isub': 1, 'iadd': 1, 'istore 0x07': 1, 'iinc   0x07, 0x01': 1, 'goto   0x0009': 1},
            1076: {'wide   0x84<-124>, 0x0007, 0x03e8<1000>': 1},
            1082: {'iload  0x06': 1, 'iload  0x07': 1, 'iadd': 1, 'sipush 0x07d0<2000>': 1, 'if_icmpne 0x000d': 1},
            1093: {'getstatic 0x001c<java/lang/System::out>': 1, 'iconst_0': 1, 'invokevirtual 0x0020<java/io/PrintStream::println>': 1},
            1100: {'goto   0x0010': 1},
            1103: {'getstatic 0x001c<java/lang/System::out>': 1, 'iload  0x06': 1, 'iload  0x07': 1, 'invokestatic 0x0026<java/lang/Math::min>': 1},
            1113: {'invokevirtual 0x0020<java/io/PrintStream::println>': 1},
            1116: {'return': 1},
            1295: {'aload_0': 1, 'arraylength': 1, 'istore_2': 1, 'iload_2': 1, 'iload_1': 1, 'iconst_1': 1, 'iadd': 1, 'if_icmpne 0x002c': 1},
            1305: {'iload_2': 1, 'newarray 0x0a': 1, 'astore_3': 1, 'iconst_0': 1, 'istore 0x04': 1},
            1312: {'iload  0x04': 1, 'aload_3': 1, 'arraylength': 1, 'if_icmpge 0x0011': 1},
            1319: {'aload_3': 1, 'iload  0x04': 2, 'aload_0': 1, 'iaload': 1, 'iastore': 1, 'iinc   0x04, 0x01': 1, 'goto   0xffee<-18>': 1},
            1333: {'getstatic 0x002a<CF::list>': 1, 'aload_3': 1, 'invokeinterface 0x0030<java/util/List::add>, 0x02, 0x00': 1},
            1342: {'pop': 1, 'goto   0x0037': 1},
            1346: {'iload_1': 1, 'istore_3': 1},
            1348: {'iload_3': 1, 'iload_2': 1, 'if_icmpge 0x0030': 1},
            1353: {'aload_0': 5, 'iload_3': 2, 'iaload': 2, 'istore 0x04': 1, 'iload_1': 3, 'iastore': 2, 'iload  0x04': 1, 'iconst_1': 1, 'iadd': 1, 'invokestatic 0x0036<CF::Permute>': 1},
            1376: {'aload_0': 4, 'iload_3': 2, 'iaload': 2, 'istore 0x05': 1, 'iload_1': 2, 'iastore': 2, 'iload  0x05': 1, 'iinc   0x03, 0x01': 1, 'goto   0xffd1<-47>': 1},
            1398: {'return': 1},
            1534: {'aload_0': 2, 'arraylength': 1, 'invokestatic 0x003a<CF::radixSort>': 1},
            1540: {'areturn': 1},
            1579: {'iload_1': 1, 'newarray 0x0a': 2, 'astore_2': 1, 'ldc    0x3e': 1, 'astore_3': 1, 'iconst_0': 1, 'istore 0x04': 1},
            1591: {'iload  0x04': 1, 'iload_1': 1, 'if_icmpge 0x0018': 1},
            1597: {'aload_3': 1, 'iconst_1': 2, 'aload_0': 1, 'iload  0x04': 1, 'iaload': 2, 'ldc    0x3f': 1, 'iand': 1, 'iadd': 2, 'dup2': 1, 'iastore': 1, 'iinc   0x04, 0x01': 1, 'goto   0xffe8<-24>': 1},
            1618: {'iconst_1': 1, 'istore 0x04': 1},
            1621: {'iload  0x04': 1, 'ldc    0x40': 1, 'if_icmpgt 0x0016': 1},
            1628: {'aload_3': 2, 'iload  0x04': 2, 'dup2': 1, 'iaload': 2, 'iconst_1': 1, 'isub': 1, 'iadd': 1, 'iastore': 1, 'iinc   0x04, 0x01': 1, 'goto   0xffe9<-23>': 1},
            1647: {'iconst_0': 1, 'istore 0x04': 1},
            1650: {'iload  0x04': 1, 'iload_1': 1, 'if_icmpge 0x001d': 1},
            1656: {'aload_2': 1, 'aload_3': 1, 'aload_0': 2, 'iload  0x04': 2, 'iaload': 3, 'ldc    0x3f': 1, 'iand': 1, 'dup2': 1, 'dup_x2': 1, 'iconst_1': 1, 'iadd': 1, 'iastore': 2, 'iinc   0x04, 0x01': 1, 'goto   0xffe3<-29>': 1},
            1682: {'aload_0': 1, 'astore 0x04': 1, 'aload_2': 1, 'astore_0': 1, 'aload  0x04': 1, 'astore_2': 1, 'ldc    0x3e': 1, 'newarray 0x0a': 1, 'astore_3': 1, 'iconst_0': 1, 'istore 0x04': 1},
            1698: {'iload  0x04': 1, 'iload_1': 1, 'if_icmpge 0x0018': 1},
            1704: {'aload_3': 1, 'iconst_1': 2, 'aload_0': 1, 'iload  0x04': 1, 'iaload': 2, 'bipush 0x10': 1, 'iushr': 1, 'iadd': 2, 'dup2': 1, 'iastore': 1, 'iinc   0x04, 0x01': 1, 'goto   0xffe8<-24>': 1},
            1725: {'iconst_1': 1, 'istore 0x04': 1},
            1728: {'iload  0x04': 1, 'ldc    0x40': 1, 'if_icmpgt 0x0016': 1},
            1735: {'aload_3': 2, 'iload  0x04': 2, 'dup2': 1, 'iaload': 2, 'iconst_1': 1, 'isub': 1, 'iadd': 1, 'iastore': 1, 'iinc   0x04, 0x01': 1, 'goto   0xffe9<-23>': 1},
            1754: {'iconst_0': 1, 'istore 0x04': 1},
            1757: {'iload  0x04': 1, 'iload_1': 1, 'if_icmpge 0x001d': 1},
            1763: {'aload_2': 1, 'aload_3': 1, 'aload_0': 2, 'iload  0x04': 2, 'iaload': 3, 'bipush 0x10': 1, 'iushr': 1, 'dup2': 1, 'dup_x2': 1, 'iconst_1': 1, 'iadd': 1, 'iastore': 2, 'iinc   0x04, 0x01': 1, 'goto   0xffe3<-29>': 1},
            1789: {'aload_0': 2, 'astore 0x04': 1, 'aload_2': 1, 'astore_0': 1, 'aload  0x04': 1, 'astore_2': 1, 'areturn': 1},
            2001: {'new    0x0041<java/util/LinkedList>': 1, 'dup': 1, 'invokespecial 0x0043<java/util/LinkedList::<init>>': 1},
            2008: {'putstatic 0x002a<CF::list>': 1, 'return': 1},
            5216: {'aload_0': 1, 'invokespecial 0x0001<java/lang/Object::<init>>': 1},
            5220: {'aload_0': 2, 'sipush 0x0400<1024>': 1, 'newarray 0x08': 1, 'putfield 0x0007': 1, 'aload_1': 1, 'putfield 0x000d': 1, 'return': 1},
            5285: {'aload_0': 1, 'getfield 0x0011': 1, 'iconst_m1': 1, 'if_icmpne 0x000b': 1},
            5293: {'new    0x0015<java/util/InputMismatchException>': 1, 'dup': 1, 'invokespecial 0x0017<java/util/InputMismatchException::<init>>': 1},
            5300: {'athrow': 1},
            5301: {'aload_0': 2, 'getfield 0x0018': 1, 'getfield 0x0011': 1, 'if_icmplt 0x002c': 1},
            5312: {'aload_0': 4, 'iconst_0': 1, 'putfield 0x0018': 1, 'getfield 0x000d': 1, 'getfield 0x0007': 1, 'invokevirtual 0x001b<java/io/InputStream::read>': 1},
            5329: {'putfield 0x0011': 1, 'goto   0x000c': 1},
            5335: {'astore_1': 1, 'new    0x0015<java/util/InputMismatchException>': 1, 'dup': 1, 'invokespecial 0x0017<java/util/InputMismatchException::<init>>': 1},
            5343: {'athrow': 1},
            5344: {'aload_0': 1, 'getfield 0x0011': 1, 'ifgt   0x0005': 1},
            5351: {'iconst_m1': 1, 'ireturn': 1},
            5353: {'aload_0': 2, 'getfield 0x0007': 1, 'dup': 1, 'getfield 0x0018': 1, 'dup_x1': 1, 'iconst_1': 1, 'iadd': 1, 'putfield 0x0018': 1, 'baload': 1, 'ireturn': 1},
            5471: {'iload_1': 1, 'bipush 0x20': 1, 'if_icmpeq 0x001a': 1},
            5477: {'iload_1': 1, 'bipush 0x0a': 1, 'if_icmpeq 0x0014': 1},
            5483: {'iload_1': 1, 'bipush 0x0d': 1, 'if_icmpeq 0x000e': 1},
            5489: {'iload_1': 1, 'bipush 0x09': 1, 'if_icmpeq 0x0008': 1},
            5495: {'iload_1': 1, 'iconst_m1': 1, 'if_icmpne 0x0007': 1},
            5500: {'iconst_1': 1, 'goto   0x0004': 1},
            5504: {'iconst_0': 1},
            5505: {'ireturn': 1},
            5556: {'iload_1': 1, 'bipush 0x0a': 1, 'if_icmpeq 0x000e': 1},
            5562: {'iload_1': 1, 'bipush 0x0d': 1, 'if_icmpeq 0x0008': 1},
            5568: {'iload_1': 1, 'iconst_m1': 1, 'if_icmpne 0x0007': 1},
            5573: {'iconst_1': 1, 'goto   0x0004': 1},
            5577: {'iconst_0': 1},
            5578: {'ireturn': 1},
            5629: {'aload_0': 1, 'invokevirtual 0x0023<CF$FastScanner::next>': 1},
            5633: {'invokestatic 0x0027<java/lang/Integer::parseInt>': 1},
            5636: {'ireturn': 1},
            5675: {'iload_1': 1, 'newarray 0x0a': 1, 'astore_2': 1, 'iconst_0': 1, 'istore_3': 1},
            5681: {'iload_3': 1, 'iload_1': 1, 'if_icmpge 0x0010': 1},
            5686: {'aload_2': 1, 'iload_3': 1, 'aload_0': 1, 'invokevirtual 0x002d<CF$FastScanner::nextInt>': 1},
            5692: {'iastore': 1, 'iinc   0x03, 0x01': 1, 'goto   0xfff1<-15>': 1},
            5699: {'aload_2': 1, 'areturn': 1},
            5773: {'iload_1': 1, 'anewarray 0x0031': 1, 'astore_2': 1, 'iconst_0': 1, 'istore_3': 1},
            5780: {'iload_3': 1, 'iload_1': 1, 'if_icmpge 0x0010': 1},
            5785: {'aload_2': 1, 'iload_3': 1, 'aload_0': 1, 'invokevirtual 0x0023<CF$FastScanner::next>': 1},
            5791: {'aastore': 1, 'iinc   0x03, 0x01': 1, 'goto   0xfff1<-15>': 1},
            5798: {'aload_2': 1, 'areturn': 1},
            5872: {'aload_0': 1, 'invokevirtual 0x0023<CF$FastScanner::next>': 1},
            5876: {'invokestatic 0x0033<java/lang/Long::parseLong>': 1},
            5879: {'lreturn': 1},
            5918: {'aload_0': 1, 'invokevirtual 0x0023<CF$FastScanner::next>': 1},
            5922: {'invokestatic 0x0039<java/lang/Double::parseDouble>': 1},
            5925: {'dreturn': 1},
            5964: {'aload_0': 1, 'invokevirtual 0x003f<CF$FastScanner::read>': 1},
            5968: {'istore_1': 1},
            5969: {'aload_0': 1, 'iload_1': 1, 'invokevirtual 0x0041<CF$FastScanner::isSpaceChar>': 1},
            5974: {'ifeq   0x000b': 1},
            5977: {'aload_0': 1, 'invokevirtual 0x003f<CF$FastScanner::read>': 1},
            5981: {'istore_1': 1, 'goto   0xfff3<-13>': 1},
            5985: {'new    0x0045<java/lang/StringBuilder>': 1, 'dup': 1, 'invokespecial 0x0047<java/lang/StringBuilder::<init>>': 1},
            5992: {'astore_2': 1},
            5993: {'aload_2': 1, 'iload_1': 1, 'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>': 1},
            5998: {'pop': 1, 'aload_0': 1, 'invokevirtual 0x003f<CF$FastScanner::read>': 1},
            6003: {'istore_1': 1, 'aload_0': 1, 'iload_1': 1, 'invokevirtual 0x0041<CF$FastScanner::isSpaceChar>': 1},
            6009: {'ifeq   0xfff0<-16>': 1},
            6012: {'aload_2': 1, 'invokevirtual 0x004c<java/lang/StringBuilder::toString>': 1},
            6016: {'areturn': 1},
            6102: {'aload_0': 1, 'invokevirtual 0x003f<CF$FastScanner::read>': 1},
            6106: {'istore_1': 1},
            6107: {'aload_0': 1, 'iload_1': 1, 'invokevirtual 0x004f<CF$FastScanner::isEndline>': 1},
            6112: {'ifeq   0x000b': 1},
            6115: {'aload_0': 1, 'invokevirtual 0x003f<CF$FastScanner::read>': 1},
            6119: {'istore_1': 1, 'goto   0xfff3<-13>': 1},
            6123: {'new    0x0045<java/lang/StringBuilder>': 1, 'dup': 1, 'invokespecial 0x0047<java/lang/StringBuilder::<init>>': 1},
            6130: {'astore_2': 1},
            6131: {'aload_2': 1, 'iload_1': 1, 'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>': 1},
            6136: {'pop': 1, 'aload_0': 1, 'invokevirtual 0x003f<CF$FastScanner::read>': 1},
            6141: {'istore_1': 1, 'aload_0': 1, 'iload_1': 1, 'invokevirtual 0x004f<CF$FastScanner::isEndline>': 1},
            6147: {'ifeq   0xfff0<-16>': 1},
            6150: {'aload_2': 1, 'invokevirtual 0x004c<java/lang/StringBuilder::toString>': 1},
            6154: {'areturn': 1},
            18446744073709508607: {},
            18446744073709511679: {},
            18446744073709512703: {},
            18446744073709524991: {},
            18446744073709527039: {},
            18446744073709531135: {},
            18446744073709533183: {},
            18446744073709534207: {},
            18446744073709536255: {},
            18446744073709537279: {},
            18446744073709540351: {},
            18446744073709541375: {},
            18446744073709542399: {},
            18446744073709544447: {},
        },
        'asm_counts_per_function': {
            902: {
                'aload_0': 1,
                'invokespecial 0x0001<java/lang/Object::<init>>': 1,
                'return': 1,
            },
            945: {
                'new    0x0007<CF$FastScanner>': 1,
                'dup': 1,
                'getstatic 0x0009<java/lang/System::in>': 1,
                'invokespecial 0x000f<CF$FastScanner::<init>>': 1,
                'astore_1': 1,
                'aload_1': 4,
                'invokevirtual 0x0012<CF$FastScanner::nextInt>': 4,
                'istore_2': 1,
                'istore_3': 1,
                'istore 0x04': 1,
                'istore 0x05': 1,
                'iconst_0': 3,
                'istore 0x06': 3,
                'iload  0x04': 5,
                'iconst_1': 2,
                'if_icmple 0x0028': 1,
                'iload  0x06': 4,
                'iload_3': 2,
                'isub': 4,
                'invokestatic 0x0016<java/lang/Math::abs>': 2,
                'wide   0x84<-124>, 0x0006, 0x03e8<1000>': 1,
                'iadd': 5,
                'iinc   0x06, 0x01': 2,
                'iload  0x05': 5,
                'iload_2': 2,
                'if_icmpge 0x0019': 1,
                'goto   0x0009': 2,
                'istore 0x07': 3,
                'if_icmpge 0x0028': 1,
                'iload  0x07': 4,
                'wide   0x84<-124>, 0x0007, 0x03e8<1000>': 1,
                'iinc   0x07, 0x01': 2,
                'if_icmple 0x0019': 1,
                'sipush 0x07d0<2000>': 1,
                'if_icmpne 0x000d': 1,
                'getstatic 0x001c<java/lang/System::out>': 2,
                'invokevirtual 0x0020<java/io/PrintStream::println>': 2,
                'invokestatic 0x0026<java/lang/Math::min>': 1,
                'goto   0x0010': 1,
                'return': 1,
            },
            1295: {
                'aload_0': 11,
                'arraylength': 2,
                'istore_2': 1,
                'iload_2': 3,
                'iload_1': 7,
                'iconst_1': 2,
                'iadd': 2,
                'if_icmpne 0x002c': 1,
                'newarray 0x0a': 1,
                'astore_3': 1,
                'iconst_0': 1,
                'istore 0x04': 2,
                'istore_3': 1,
                'iload  0x04': 4,
                'aload_3': 3,
                'if_icmpge 0x0011': 1,
                'iaload': 5,
                'iastore': 5,
                'iinc   0x04, 0x01': 1,
                'goto   0xffee<-18>': 1,
                'getstatic 0x002a<CF::list>': 1,
                'invokeinterface 0x0030<java/util/List::add>, 0x02, 0x00': 1,
                'pop': 1,
                'goto   0x0037': 1,
                'return': 1,
                'iload_3': 5,
                'if_icmpge 0x0030': 1,
                'invokestatic 0x0036<CF::Permute>': 1,
                'istore 0x05': 1,
                'iload  0x05': 1,
                'iinc   0x03, 0x01': 1,
                'goto   0xffd1<-47>': 1,
            },
            1534: {
                'aload_0': 2,
                'arraylength': 1,
                'invokestatic 0x003a<CF::radixSort>': 1,
                'areturn': 1,
            },
            1579: {
                'iload_1': 5,
                'newarray 0x0a': 3,
                'astore_2': 3,
                'ldc    0x3e': 2,
                'astore_3': 2,
                'iconst_0': 4,
                'istore 0x04': 6,
                'iload  0x04': 16,
                'if_icmpge 0x0018': 2,
                'aload_3': 8,
                'iconst_1': 10,
                'aload_0': 9,
                'iaload': 14,
                'ldc    0x3f': 2,
                'iand': 2,
                'iadd': 8,
                'dup2': 6,
                'iastore': 8,
                'iinc   0x04, 0x01': 6,
                'goto   0xffe8<-24>': 2,
                'ldc    0x40': 2,
                'if_icmpgt 0x0016': 2,
                'isub': 2,
                'goto   0xffe9<-23>': 2,
                'if_icmpge 0x001d': 2,
                'aload_2': 4,
                'dup_x2': 2,
                'goto   0xffe3<-29>': 2,
                'astore 0x04': 2,
                'astore_0': 2,
                'aload  0x04': 2,
                'bipush 0x10': 2,
                'iushr': 2,
                'areturn': 1,
            },
            2001: {
                'new    0x0041<java/util/LinkedList>': 1,
                'dup': 1,
                'invokespecial 0x0043<java/util/LinkedList::<init>>': 1,
                'putstatic 0x002a<CF::list>': 1,
                'return': 1,
            },
            5216: {
                'aload_0': 3,
                'invokespecial 0x0001<java/lang/Object::<init>>': 1,
                'sipush 0x0400<1024>': 1,
                'newarray 0x08': 1,
                'putfield 0x0007': 1,
                'aload_1': 1,
                'putfield 0x000d': 1,
                'return': 1,
            },
            5285: {
                'aload_0': 10,
                'getfield 0x0011': 3,
                'iconst_m1': 2,
                'if_icmpne 0x000b': 1,
                'new    0x0015<java/util/InputMismatchException>': 2,
                'dup': 3,
                'invokespecial 0x0017<java/util/InputMismatchException::<init>>': 2,
                'getfield 0x0018': 2,
                'if_icmplt 0x002c': 1,
                'athrow': 2,
                'iconst_0': 1,
                'putfield 0x0018': 2,
                'getfield 0x000d': 1,
                'getfield 0x0007': 2,
                'invokevirtual 0x001b<java/io/InputStream::read>': 1,
                'dup_x1': 1,
                'iconst_1': 1,
                'iadd': 1,
                'baload': 1,
                'ireturn': 2,
                'putfield 0x0011': 1,
                'goto   0x000c': 1,
                'ifgt   0x0005': 1,
                'astore_1': 1,
            },
            5471: {
                'iload_1': 5,
                'bipush 0x20': 1,
                'if_icmpeq 0x001a': 1,
                'bipush 0x0a': 1,
                'if_icmpeq 0x0014': 1,
                'iconst_1': 1,
                'goto   0x0004': 1,
                'bipush 0x0d': 1,
                'if_icmpeq 0x000e': 1,
                'bipush 0x09': 1,
                'if_icmpeq 0x0008': 1,
                'iconst_m1': 1,
                'if_icmpne 0x0007': 1,
                'iconst_0': 1,
                'ireturn': 1,
            },
            5556: {
                'iload_1': 3,
                'bipush 0x0a': 1,
                'if_icmpeq 0x000e': 1,
                'bipush 0x0d': 1,
                'if_icmpeq 0x0008': 1,
                'iconst_1': 1,
                'goto   0x0004': 1,
                'iconst_m1': 1,
                'if_icmpne 0x0007': 1,
                'iconst_0': 1,
                'ireturn': 1,
            },
            5629: {
                'aload_0': 1,
                'invokevirtual 0x0023<CF$FastScanner::next>': 1,
                'invokestatic 0x0027<java/lang/Integer::parseInt>': 1,
                'ireturn': 1,
            },
            5675: {
                'iload_1': 2,
                'newarray 0x0a': 1,
                'astore_2': 1,
                'iconst_0': 1,
                'istore_3': 1,
                'iload_3': 2,
                'if_icmpge 0x0010': 1,
                'aload_2': 2,
                'aload_0': 1,
                'invokevirtual 0x002d<CF$FastScanner::nextInt>': 1,
                'areturn': 1,
                'iastore': 1,
                'iinc   0x03, 0x01': 1,
                'goto   0xfff1<-15>': 1,
            },
            5773: {
                'iload_1': 2,
                'anewarray 0x0031': 1,
                'astore_2': 1,
                'iconst_0': 1,
                'istore_3': 1,
                'iload_3': 2,
                'if_icmpge 0x0010': 1,
                'aload_2': 2,
                'aload_0': 1,
                'invokevirtual 0x0023<CF$FastScanner::next>': 1,
                'areturn': 1,
                'aastore': 1,
                'iinc   0x03, 0x01': 1,
                'goto   0xfff1<-15>': 1,
            },
            5872: {
                'aload_0': 1,
                'invokevirtual 0x0023<CF$FastScanner::next>': 1,
                'invokestatic 0x0033<java/lang/Long::parseLong>': 1,
                'lreturn': 1,
            },
            5918: {
                'aload_0': 1,
                'invokevirtual 0x0023<CF$FastScanner::next>': 1,
                'invokestatic 0x0039<java/lang/Double::parseDouble>': 1,
                'dreturn': 1,
            },
            5964: {
                'aload_0': 5,
                'invokevirtual 0x003f<CF$FastScanner::read>': 3,
                'istore_1': 3,
                'iload_1': 3,
                'invokevirtual 0x0041<CF$FastScanner::isSpaceChar>': 2,
                'ifeq   0x000b': 1,
                'new    0x0045<java/lang/StringBuilder>': 1,
                'dup': 1,
                'invokespecial 0x0047<java/lang/StringBuilder::<init>>': 1,
                'goto   0xfff3<-13>': 1,
                'astore_2': 1,
                'aload_2': 2,
                'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>': 1,
                'pop': 1,
                'ifeq   0xfff0<-16>': 1,
                'invokevirtual 0x004c<java/lang/StringBuilder::toString>': 1,
                'areturn': 1,
            },
            6102: {
                'aload_0': 5,
                'invokevirtual 0x003f<CF$FastScanner::read>': 3,
                'istore_1': 3,
                'iload_1': 3,
                'invokevirtual 0x004f<CF$FastScanner::isEndline>': 2,
                'ifeq   0x000b': 1,
                'new    0x0045<java/lang/StringBuilder>': 1,
                'dup': 1,
                'invokespecial 0x0047<java/lang/StringBuilder::<init>>': 1,
                'goto   0xfff3<-13>': 1,
                'astore_2': 1,
                'aload_2': 2,
                'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>': 1,
                'pop': 1,
                'ifeq   0xfff0<-16>': 1,
                'invokevirtual 0x004c<java/lang/StringBuilder::toString>': 1,
                'areturn': 1,
            },
            18446744073709508607: {
                
            },
            18446744073709511679: {
                
            },
            18446744073709512703: {
                
            },
            18446744073709524991: {
                
            },
            18446744073709527039: {
                
            },
            18446744073709531135: {
                
            },
            18446744073709533183: {
                
            },
            18446744073709534207: {
                
            },
            18446744073709536255: {
                
            },
            18446744073709537279: {
                
            },
            18446744073709540351: {
                
            },
            18446744073709541375: {
                
            },
            18446744073709542399: {
                
            },
            18446744073709544447: {
                
            },
        },
        'asm_counts': {
            'aload_0': 51,
            'invokespecial 0x0001<java/lang/Object::<init>>': 2,
            'return': 5,
            'new    0x0007<CF$FastScanner>': 1,
            'dup': 7,
            'getstatic 0x0009<java/lang/System::in>': 1,
            'invokespecial 0x000f<CF$FastScanner::<init>>': 1,
            'astore_1': 2,
            'aload_1': 5,
            'invokevirtual 0x0012<CF$FastScanner::nextInt>': 4,
            'istore_2': 2,
            'istore_3': 4,
            'istore 0x04': 9,
            'istore 0x05': 2,
            'iconst_0': 13,
            'istore 0x06': 3,
            'iload  0x04': 25,
            'iconst_1': 17,
            'if_icmple 0x0028': 1,
            'iload  0x06': 4,
            'iload_3': 11,
            'isub': 6,
            'invokestatic 0x0016<java/lang/Math::abs>': 2,
            'wide   0x84<-124>, 0x0006, 0x03e8<1000>': 1,
            'iadd': 16,
            'iinc   0x06, 0x01': 2,
            'iload  0x05': 6,
            'iload_2': 5,
            'if_icmpge 0x0019': 1,
            'goto   0x0009': 2,
            'istore 0x07': 3,
            'if_icmpge 0x0028': 1,
            'iload  0x07': 4,
            'wide   0x84<-124>, 0x0007, 0x03e8<1000>': 1,
            'iinc   0x07, 0x01': 2,
            'if_icmple 0x0019': 1,
            'sipush 0x07d0<2000>': 1,
            'if_icmpne 0x000d': 1,
            'getstatic 0x001c<java/lang/System::out>': 2,
            'invokevirtual 0x0020<java/io/PrintStream::println>': 2,
            'invokestatic 0x0026<java/lang/Math::min>': 1,
            'goto   0x0010': 1,
            'arraylength': 3,
            'iload_1': 30,
            'if_icmpne 0x002c': 1,
            'newarray 0x0a': 5,
            'astore_3': 3,
            'aload_3': 11,
            'if_icmpge 0x0011': 1,
            'iaload': 19,
            'iastore': 14,
            'iinc   0x04, 0x01': 7,
            'goto   0xffee<-18>': 1,
            'getstatic 0x002a<CF::list>': 1,
            'invokeinterface 0x0030<java/util/List::add>, 0x02, 0x00': 1,
            'pop': 3,
            'goto   0x0037': 1,
            'if_icmpge 0x0030': 1,
            'invokestatic 0x0036<CF::Permute>': 1,
            'iinc   0x03, 0x01': 3,
            'goto   0xffd1<-47>': 1,
            'invokestatic 0x003a<CF::radixSort>': 1,
            'areturn': 6,
            'astore_2': 7,
            'ldc    0x3e': 2,
            'if_icmpge 0x0018': 2,
            'ldc    0x3f': 2,
            'iand': 2,
            'dup2': 6,
            'goto   0xffe8<-24>': 2,
            'ldc    0x40': 2,
            'if_icmpgt 0x0016': 2,
            'goto   0xffe9<-23>': 2,
            'if_icmpge 0x001d': 2,
            'aload_2': 12,
            'dup_x2': 2,
            'goto   0xffe3<-29>': 2,
            'astore 0x04': 2,
            'astore_0': 2,
            'aload  0x04': 2,
            'bipush 0x10': 2,
            'iushr': 2,
            'new    0x0041<java/util/LinkedList>': 1,
            'invokespecial 0x0043<java/util/LinkedList::<init>>': 1,
            'putstatic 0x002a<CF::list>': 1,
            'sipush 0x0400<1024>': 1,
            'newarray 0x08': 1,
            'putfield 0x0007': 1,
            'putfield 0x000d': 1,
            'getfield 0x0011': 3,
            'iconst_m1': 4,
            'if_icmpne 0x000b': 1,
            'new    0x0015<java/util/InputMismatchException>': 2,
            'invokespecial 0x0017<java/util/InputMismatchException::<init>>': 2,
            'getfield 0x0018': 2,
            'if_icmplt 0x002c': 1,
            'athrow': 2,
            'putfield 0x0018': 2,
            'getfield 0x000d': 1,
            'getfield 0x0007': 2,
            'invokevirtual 0x001b<java/io/InputStream::read>': 1,
            'dup_x1': 1,
            'baload': 1,
            'ireturn': 5,
            'putfield 0x0011': 1,
            'goto   0x000c': 1,
            'ifgt   0x0005': 1,
            'bipush 0x20': 1,
            'if_icmpeq 0x001a': 1,
            'bipush 0x0a': 2,
            'if_icmpeq 0x0014': 1,
            'goto   0x0004': 2,
            'bipush 0x0d': 2,
            'if_icmpeq 0x000e': 2,
            'bipush 0x09': 1,
            'if_icmpeq 0x0008': 2,
            'if_icmpne 0x0007': 2,
            'invokevirtual 0x0023<CF$FastScanner::next>': 4,
            'invokestatic 0x0027<java/lang/Integer::parseInt>': 1,
            'if_icmpge 0x0010': 2,
            'invokevirtual 0x002d<CF$FastScanner::nextInt>': 1,
            'goto   0xfff1<-15>': 2,
            'anewarray 0x0031': 1,
            'aastore': 1,
            'invokestatic 0x0033<java/lang/Long::parseLong>': 1,
            'lreturn': 1,
            'invokestatic 0x0039<java/lang/Double::parseDouble>': 1,
            'dreturn': 1,
            'invokevirtual 0x003f<CF$FastScanner::read>': 6,
            'istore_1': 6,
            'invokevirtual 0x0041<CF$FastScanner::isSpaceChar>': 2,
            'ifeq   0x000b': 2,
            'new    0x0045<java/lang/StringBuilder>': 2,
            'invokespecial 0x0047<java/lang/StringBuilder::<init>>': 2,
            'goto   0xfff3<-13>': 2,
            'invokevirtual 0x0048<java/lang/StringBuilder::appendCodePoint>': 2,
            'ifeq   0xfff0<-16>': 2,
            'invokevirtual 0x004c<java/lang/StringBuilder::toString>': 2,
            'invokevirtual 0x004f<CF$FastScanner::isEndline>': 2,
        },
    }


    # Adding functions to the cfg
    if build_level in ['cfg']:
        __auto_cfg.add_function(*__auto_functions.values())
    else:
        __auto_cfg.functions = list(__auto_functions.values())
        __auto_cfg.blocks = list(__auto_blocks.values())

    return {
        'blocks': __auto_blocks,
        'file': os.path.basename(__file__),
        'inputs': [_MANUAL_ROSE_GV_STR],
        'cfg': __auto_cfg,
        'functions': __auto_functions,
        'expected': expected,
    }


_MANUAL_ROSE_GV_STR = """digraph CFG {
 graph [ overlap=scale ];
 node  [  ];
 edge  [  ];

subgraph cluster_0x00000386 { label="function 0x00000386 \\"CF::<init>\\"" fillcolor="#f2f2f2" href="0x00000386" style=filled
V_0x00000386 [ label=<00000386  ?? aload_0 <br align="left"/>00000387  ?? invokespecial 0x0001&lt;java/lang/Object::&lt;init&gt;&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00000386" shape=box style=filled ];
V_0x0000038a [ label=<0000038a  ?? return <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x0000038a" shape=box style=filled ];
}

subgraph cluster_0x000003b1 { label="function 0x000003b1 \\"CF::main\\"" fillcolor="#f2f2f2" href="0x000003b1" style=filled
V_0x000003b1 [ label=<000003b1  ?? new    0x0007&lt;CF$FastScanner&gt;<br align="left"/>000003b4  ?? dup    <br align="left"/>000003b5  ?? getstatic 0x0009&lt;java/lang/System::in&gt;<br align="left"/>000003b8  ?? invokespecial 0x000f&lt;CF$FastScanner::&lt;init&gt;&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000003b1" shape=box style=filled ];
V_0x000003bb [ label=<000003bb  ?? astore_1 <br align="left"/>000003bc  ?? aload_1 <br align="left"/>000003bd  ?? invokevirtual 0x0012&lt;CF$FastScanner::nextInt&gt;<br align="left"/>> fontname=Courier href="0x000003bb" shape=box ];
V_0x000003c0 [ label=<000003c0  ?? istore_2 <br align="left"/>000003c1  ?? aload_1 <br align="left"/>000003c2  ?? invokevirtual 0x0012&lt;CF$FastScanner::nextInt&gt;<br align="left"/>> fontname=Courier href="0x000003c0" shape=box ];
V_0x000003c5 [ label=<000003c5  ?? istore_3 <br align="left"/>000003c6  ?? aload_1 <br align="left"/>000003c7  ?? invokevirtual 0x0012&lt;CF$FastScanner::nextInt&gt;<br align="left"/>> fontname=Courier href="0x000003c5" shape=box ];
V_0x000003ca [ label=<000003ca  ?? istore 0x04<br align="left"/>000003cc  ?? aload_1 <br align="left"/>000003cd  ?? invokevirtual 0x0012&lt;CF$FastScanner::nextInt&gt;<br align="left"/>> fontname=Courier href="0x000003ca" shape=box ];
V_0x000003d0 [ label=<000003d0  ?? istore 0x05<br align="left"/>000003d2  ?? iconst_0 <br align="left"/>000003d3  ?? istore 0x06<br align="left"/>000003d5  ?? iload  0x04<br align="left"/>000003d7  ?? iconst_1 <br align="left"/>000003d8  ?? if_icmple 0x0028<br align="left"/>> fontname=Courier href="0x000003d0" shape=box ];
V_0x000003db [ label=<000003db  ?? iload  0x06<br align="left"/>000003dd  ?? iload_3 <br align="left"/>000003de  ?? iload  0x04<br align="left"/>000003e0  ?? isub   <br align="left"/>000003e1  ?? invokestatic 0x0016&lt;java/lang/Math::abs&gt;<br align="left"/>> fontname=Courier href="0x000003db" shape=box ];
V_0x00000400 [ label=<00000400  ?? wide   0x84&lt;-124&gt;, 0x0006, 0x03e8&lt;1000&gt;<br align="left"/>> fontname=Courier href="0x00000400" shape=box ];
V_0x000003e4 [ label=<000003e4  ?? iadd   <br align="left"/>000003e5  ?? istore 0x06<br align="left"/>000003e7  ?? iinc   0x06, 0x01<br align="left"/>000003ea  ?? iload  0x05<br align="left"/>000003ec  ?? iload_2 <br align="left"/>000003ed  ?? if_icmpge 0x0019<br align="left"/>> fontname=Courier href="0x000003e4" shape=box ];
V_0x000003f0 [ label=<000003f0  ?? iload  0x06<br align="left"/>000003f2  ?? iload  0x05<br align="left"/>000003f4  ?? iload  0x04<br align="left"/>000003f6  ?? isub   <br align="left"/>000003f7  ?? iadd   <br align="left"/>000003f8  ?? istore 0x06<br align="left"/>000003fa  ?? iinc   0x06, 0x01<br align="left"/>000003fd  ?? goto   0x0009<br align="left"/>> fontname=Courier href="0x000003f0" shape=box ];
V_0x00000406 [ label=<00000406  ?? iconst_0 <br align="left"/>00000407  ?? istore 0x07<br align="left"/>00000409  ?? iload  0x05<br align="left"/>0000040b  ?? iload_2 <br align="left"/>0000040c  ?? if_icmpge 0x0028<br align="left"/>> fontname=Courier href="0x00000406" shape=box ];
V_0x0000040f [ label=<0000040f  ?? iload  0x07<br align="left"/>00000411  ?? iload_3 <br align="left"/>00000412  ?? iload  0x05<br align="left"/>00000414  ?? isub   <br align="left"/>00000415  ?? invokestatic 0x0016&lt;java/lang/Math::abs&gt;<br align="left"/>> fontname=Courier href="0x0000040f" shape=box ];
V_0x00000434 [ label=<00000434  ?? wide   0x84&lt;-124&gt;, 0x0007, 0x03e8&lt;1000&gt;<br align="left"/>> fontname=Courier href="0x00000434" shape=box ];
V_0x00000418 [ label=<00000418  ?? iadd   <br align="left"/>00000419  ?? istore 0x07<br align="left"/>0000041b  ?? iinc   0x07, 0x01<br align="left"/>0000041e  ?? iload  0x04<br align="left"/>00000420  ?? iconst_1 <br align="left"/>00000421  ?? if_icmple 0x0019<br align="left"/>> fontname=Courier href="0x00000418" shape=box ];
V_0x00000424 [ label=<00000424  ?? iload  0x07<br align="left"/>00000426  ?? iload  0x05<br align="left"/>00000428  ?? iload  0x04<br align="left"/>0000042a  ?? isub   <br align="left"/>0000042b  ?? iadd   <br align="left"/>0000042c  ?? istore 0x07<br align="left"/>0000042e  ?? iinc   0x07, 0x01<br align="left"/>00000431  ?? goto   0x0009<br align="left"/>> fontname=Courier href="0x00000424" shape=box ];
V_0x0000043a [ label=<0000043a  ?? iload  0x06<br align="left"/>0000043c  ?? iload  0x07<br align="left"/>0000043e  ?? iadd   <br align="left"/>0000043f  ?? sipush 0x07d0&lt;2000&gt;<br align="left"/>00000442  ?? if_icmpne 0x000d<br align="left"/>> fontname=Courier href="0x0000043a" shape=box ];
V_0x00000445 [ label=<00000445  ?? getstatic 0x001c&lt;java/lang/System::out&gt;<br align="left"/>00000448  ?? iconst_0 <br align="left"/>00000449  ?? invokevirtual 0x0020&lt;java/io/PrintStream::println&gt;<br align="left"/>> fontname=Courier href="0x00000445" shape=box ];
V_0x0000044f [ label=<0000044f  ?? getstatic 0x001c&lt;java/lang/System::out&gt;<br align="left"/>00000452  ?? iload  0x06<br align="left"/>00000454  ?? iload  0x07<br align="left"/>00000456  ?? invokestatic 0x0026&lt;java/lang/Math::min&gt;<br align="left"/>> fontname=Courier href="0x0000044f" shape=box ];
V_0x0000044c [ label=<0000044c  ?? goto   0x0010<br align="left"/>> fontname=Courier href="0x0000044c" shape=box ];
V_0x0000045c [ label=<0000045c  ?? return <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x0000045c" shape=box style=filled ];
V_0x00000459 [ label=<00000459  ?? invokevirtual 0x0020&lt;java/io/PrintStream::println&gt;<br align="left"/>> fontname=Courier href="0x00000459" shape=box ];
}

subgraph cluster_0x0000050f { label="function 0x0000050f \\"CF::Permute\\"" fillcolor="#f2f2f2" href="0x0000050f" style=filled
V_0x0000050f [ label=<0000050f  ?? aload_0 <br align="left"/>00000510  ?? arraylength <br align="left"/>00000511  ?? istore_2 <br align="left"/>00000512  ?? iload_2 <br align="left"/>00000513  ?? iload_1 <br align="left"/>00000514  ?? iconst_1 <br align="left"/>00000515  ?? iadd   <br align="left"/>00000516  ?? if_icmpne 0x002c<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000050f" shape=box style=filled ];
V_0x00000519 [ label=<00000519  ?? iload_2 <br align="left"/>0000051a  ?? newarray 0x0a<br align="left"/>0000051c  ?? astore_3 <br align="left"/>0000051d  ?? iconst_0 <br align="left"/>0000051e  ?? istore 0x04<br align="left"/>> fontname=Courier href="0x00000519" shape=box ];
V_0x00000542 [ label=<00000542  ?? iload_1 <br align="left"/>00000543  ?? istore_3 <br align="left"/>> fontname=Courier href="0x00000542" shape=box ];
V_0x00000520 [ label=<00000520  ?? iload  0x04<br align="left"/>00000522  ?? aload_3 <br align="left"/>00000523  ?? arraylength <br align="left"/>00000524  ?? if_icmpge 0x0011<br align="left"/>> fontname=Courier href="0x00000520" shape=box ];
V_0x00000527 [ label=<00000527  ?? aload_3 <br align="left"/>00000528  ?? iload  0x04<br align="left"/>0000052a  ?? aload_0 <br align="left"/>0000052b  ?? iload  0x04<br align="left"/>0000052d  ?? iaload <br align="left"/>0000052e  ?? iastore <br align="left"/>0000052f  ?? iinc   0x04, 0x01<br align="left"/>00000532  ?? goto   0xffee&lt;-18&gt;<br align="left"/>> fontname=Courier href="0x00000527" shape=box ];
V_0x00000535 [ label=<00000535  ?? getstatic 0x002a&lt;CF::list&gt;<br align="left"/>00000538  ?? aload_3 <br align="left"/>00000539  ?? invokeinterface 0x0030&lt;java/util/List::add&gt;, 0x02, 0x00<br align="left"/>> fontname=Courier href="0x00000535" shape=box ];
V_0x0000053e [ label=<0000053e  ?? pop    <br align="left"/>0000053f  ?? goto   0x0037<br align="left"/>> fontname=Courier href="0x0000053e" shape=box ];
V_0x00000576 [ label=<00000576  ?? return <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00000576" shape=box style=filled ];
V_0x00000544 [ label=<00000544  ?? iload_3 <br align="left"/>00000545  ?? iload_2 <br align="left"/>00000546  ?? if_icmpge 0x0030<br align="left"/>> fontname=Courier href="0x00000544" shape=box ];
V_0x00000549 [ label=<00000549  ?? aload_0 <br align="left"/>0000054a  ?? iload_3 <br align="left"/>0000054b  ?? iaload <br align="left"/>0000054c  ?? istore 0x04<br align="left"/>0000054e  ?? aload_0 <br align="left"/>0000054f  ?? iload_3 <br align="left"/>00000550  ?? aload_0 <br align="left"/>00000551  ?? iload_1 <br align="left"/>00000552  ?? iaload <br align="left"/>00000553  ?? iastore <br align="left"/>00000554  ?? aload_0 <br align="left"/>00000555  ?? iload_1 <br align="left"/>00000556  ?? iload  0x04<br align="left"/>00000558  ?? iastore <br align="left"/>00000559  ?? aload_0 <br align="left"/>0000055a  ?? iload_1 <br align="left"/>0000055b  ?? iconst_1 <br align="left"/>0000055c  ?? iadd   <br align="left"/>0000055d  ?? invokestatic 0x0036&lt;CF::Permute&gt;<br align="left"/>> fontname=Courier href="0x00000549" shape=box ];
V_0x00000560 [ label=<00000560  ?? aload_0 <br align="left"/>00000561  ?? iload_3 <br align="left"/>00000562  ?? iaload <br align="left"/>00000563  ?? istore 0x05<br align="left"/>00000565  ?? aload_0 <br align="left"/>00000566  ?? iload_3 <br align="left"/>00000567  ?? aload_0 <br align="left"/>00000568  ?? iload_1 <br align="left"/>00000569  ?? iaload <br align="left"/>0000056a  ?? iastore <br align="left"/>0000056b  ?? aload_0 <br align="left"/>0000056c  ?? iload_1 <br align="left"/>0000056d  ?? iload  0x05<br align="left"/>0000056f  ?? iastore <br align="left"/>00000570  ?? iinc   0x03, 0x01<br align="left"/>00000573  ?? goto   0xffd1&lt;-47&gt;<br align="left"/>> fontname=Courier href="0x00000560" shape=box ];
}

subgraph cluster_0x000005fe { label="function 0x000005fe \\"CF::radixSort\\"" fillcolor="#f2f2f2" href="0x000005fe" style=filled
V_0x000005fe [ label=<000005fe  ?? aload_0 <br align="left"/>000005ff  ?? aload_0 <br align="left"/>00000600  ?? arraylength <br align="left"/>00000601  ?? invokestatic 0x003a&lt;CF::radixSort&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000005fe" shape=box style=filled ];
V_0x00000604 [ label=<00000604  ?? areturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00000604" shape=box style=filled ];
}

subgraph cluster_0x0000062b { label="function 0x0000062b \\"CF::radixSort\\"" fillcolor="#f2f2f2" href="0x0000062b" style=filled
V_0x0000062b [ label=<0000062b  ?? iload_1 <br align="left"/>0000062c  ?? newarray 0x0a<br align="left"/>0000062e  ?? astore_2 <br align="left"/>0000062f  ?? ldc    0x3e<br align="left"/>00000631  ?? newarray 0x0a<br align="left"/>00000633  ?? astore_3 <br align="left"/>00000634  ?? iconst_0 <br align="left"/>00000635  ?? istore 0x04<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000062b" shape=box style=filled ];
V_0x00000637 [ label=<00000637  ?? iload  0x04<br align="left"/>00000639  ?? iload_1 <br align="left"/>0000063a  ?? if_icmpge 0x0018<br align="left"/>> fontname=Courier href="0x00000637" shape=box ];
V_0x0000063d [ label=<0000063d  ?? aload_3 <br align="left"/>0000063e  ?? iconst_1 <br align="left"/>0000063f  ?? aload_0 <br align="left"/>00000640  ?? iload  0x04<br align="left"/>00000642  ?? iaload <br align="left"/>00000643  ?? ldc    0x3f<br align="left"/>00000645  ?? iand   <br align="left"/>00000646  ?? iadd   <br align="left"/>00000647  ?? dup2   <br align="left"/>00000648  ?? iaload <br align="left"/>00000649  ?? iconst_1 <br align="left"/>0000064a  ?? iadd   <br align="left"/>0000064b  ?? iastore <br align="left"/>0000064c  ?? iinc   0x04, 0x01<br align="left"/>0000064f  ?? goto   0xffe8&lt;-24&gt;<br align="left"/>> fontname=Courier href="0x0000063d" shape=box ];
V_0x00000652 [ label=<00000652  ?? iconst_1 <br align="left"/>00000653  ?? istore 0x04<br align="left"/>> fontname=Courier href="0x00000652" shape=box ];
V_0x00000655 [ label=<00000655  ?? iload  0x04<br align="left"/>00000657  ?? ldc    0x40<br align="left"/>00000659  ?? if_icmpgt 0x0016<br align="left"/>> fontname=Courier href="0x00000655" shape=box ];
V_0x0000065c [ label=<0000065c  ?? aload_3 <br align="left"/>0000065d  ?? iload  0x04<br align="left"/>0000065f  ?? dup2   <br align="left"/>00000660  ?? iaload <br align="left"/>00000661  ?? aload_3 <br align="left"/>00000662  ?? iload  0x04<br align="left"/>00000664  ?? iconst_1 <br align="left"/>00000665  ?? isub   <br align="left"/>00000666  ?? iaload <br align="left"/>00000667  ?? iadd   <br align="left"/>00000668  ?? iastore <br align="left"/>00000669  ?? iinc   0x04, 0x01<br align="left"/>0000066c  ?? goto   0xffe9&lt;-23&gt;<br align="left"/>> fontname=Courier href="0x0000065c" shape=box ];
V_0x0000066f [ label=<0000066f  ?? iconst_0 <br align="left"/>00000670  ?? istore 0x04<br align="left"/>> fontname=Courier href="0x0000066f" shape=box ];
V_0x00000672 [ label=<00000672  ?? iload  0x04<br align="left"/>00000674  ?? iload_1 <br align="left"/>00000675  ?? if_icmpge 0x001d<br align="left"/>> fontname=Courier href="0x00000672" shape=box ];
V_0x00000678 [ label=<00000678  ?? aload_2 <br align="left"/>00000679  ?? aload_3 <br align="left"/>0000067a  ?? aload_0 <br align="left"/>0000067b  ?? iload  0x04<br align="left"/>0000067d  ?? iaload <br align="left"/>0000067e  ?? ldc    0x3f<br align="left"/>00000680  ?? iand   <br align="left"/>00000681  ?? dup2   <br align="left"/>00000682  ?? iaload <br align="left"/>00000683  ?? dup_x2 <br align="left"/>00000684  ?? iconst_1 <br align="left"/>00000685  ?? iadd   <br align="left"/>00000686  ?? iastore <br align="left"/>00000687  ?? aload_0 <br align="left"/>00000688  ?? iload  0x04<br align="left"/>0000068a  ?? iaload <br align="left"/>0000068b  ?? iastore <br align="left"/>0000068c  ?? iinc   0x04, 0x01<br align="left"/>0000068f  ?? goto   0xffe3&lt;-29&gt;<br align="left"/>> fontname=Courier href="0x00000678" shape=box ];
V_0x00000692 [ label=<00000692  ?? aload_0 <br align="left"/>00000693  ?? astore 0x04<br align="left"/>00000695  ?? aload_2 <br align="left"/>00000696  ?? astore_0 <br align="left"/>00000697  ?? aload  0x04<br align="left"/>00000699  ?? astore_2 <br align="left"/>0000069a  ?? ldc    0x3e<br align="left"/>0000069c  ?? newarray 0x0a<br align="left"/>0000069e  ?? astore_3 <br align="left"/>0000069f  ?? iconst_0 <br align="left"/>000006a0  ?? istore 0x04<br align="left"/>> fontname=Courier href="0x00000692" shape=box ];
V_0x000006a2 [ label=<000006a2  ?? iload  0x04<br align="left"/>000006a4  ?? iload_1 <br align="left"/>000006a5  ?? if_icmpge 0x0018<br align="left"/>> fontname=Courier href="0x000006a2" shape=box ];
V_0x000006a8 [ label=<000006a8  ?? aload_3 <br align="left"/>000006a9  ?? iconst_1 <br align="left"/>000006aa  ?? aload_0 <br align="left"/>000006ab  ?? iload  0x04<br align="left"/>000006ad  ?? iaload <br align="left"/>000006ae  ?? bipush 0x10<br align="left"/>000006b0  ?? iushr  <br align="left"/>000006b1  ?? iadd   <br align="left"/>000006b2  ?? dup2   <br align="left"/>000006b3  ?? iaload <br align="left"/>000006b4  ?? iconst_1 <br align="left"/>000006b5  ?? iadd   <br align="left"/>000006b6  ?? iastore <br align="left"/>000006b7  ?? iinc   0x04, 0x01<br align="left"/>000006ba  ?? goto   0xffe8&lt;-24&gt;<br align="left"/>> fontname=Courier href="0x000006a8" shape=box ];
V_0x000006bd [ label=<000006bd  ?? iconst_1 <br align="left"/>000006be  ?? istore 0x04<br align="left"/>> fontname=Courier href="0x000006bd" shape=box ];
V_0x000006c0 [ label=<000006c0  ?? iload  0x04<br align="left"/>000006c2  ?? ldc    0x40<br align="left"/>000006c4  ?? if_icmpgt 0x0016<br align="left"/>> fontname=Courier href="0x000006c0" shape=box ];
V_0x000006c7 [ label=<000006c7  ?? aload_3 <br align="left"/>000006c8  ?? iload  0x04<br align="left"/>000006ca  ?? dup2   <br align="left"/>000006cb  ?? iaload <br align="left"/>000006cc  ?? aload_3 <br align="left"/>000006cd  ?? iload  0x04<br align="left"/>000006cf  ?? iconst_1 <br align="left"/>000006d0  ?? isub   <br align="left"/>000006d1  ?? iaload <br align="left"/>000006d2  ?? iadd   <br align="left"/>000006d3  ?? iastore <br align="left"/>000006d4  ?? iinc   0x04, 0x01<br align="left"/>000006d7  ?? goto   0xffe9&lt;-23&gt;<br align="left"/>> fontname=Courier href="0x000006c7" shape=box ];
V_0x000006da [ label=<000006da  ?? iconst_0 <br align="left"/>000006db  ?? istore 0x04<br align="left"/>> fontname=Courier href="0x000006da" shape=box ];
V_0x000006dd [ label=<000006dd  ?? iload  0x04<br align="left"/>000006df  ?? iload_1 <br align="left"/>000006e0  ?? if_icmpge 0x001d<br align="left"/>> fontname=Courier href="0x000006dd" shape=box ];
V_0x000006e3 [ label=<000006e3  ?? aload_2 <br align="left"/>000006e4  ?? aload_3 <br align="left"/>000006e5  ?? aload_0 <br align="left"/>000006e6  ?? iload  0x04<br align="left"/>000006e8  ?? iaload <br align="left"/>000006e9  ?? bipush 0x10<br align="left"/>000006eb  ?? iushr  <br align="left"/>000006ec  ?? dup2   <br align="left"/>000006ed  ?? iaload <br align="left"/>000006ee  ?? dup_x2 <br align="left"/>000006ef  ?? iconst_1 <br align="left"/>000006f0  ?? iadd   <br align="left"/>000006f1  ?? iastore <br align="left"/>000006f2  ?? aload_0 <br align="left"/>000006f3  ?? iload  0x04<br align="left"/>000006f5  ?? iaload <br align="left"/>000006f6  ?? iastore <br align="left"/>000006f7  ?? iinc   0x04, 0x01<br align="left"/>000006fa  ?? goto   0xffe3&lt;-29&gt;<br align="left"/>> fontname=Courier href="0x000006e3" shape=box ];
V_0x000006fd [ label=<000006fd  ?? aload_0 <br align="left"/>000006fe  ?? astore 0x04<br align="left"/>00000700  ?? aload_2 <br align="left"/>00000701  ?? astore_0 <br align="left"/>00000702  ?? aload  0x04<br align="left"/>00000704  ?? astore_2 <br align="left"/>00000705  ?? aload_0 <br align="left"/>00000706  ?? areturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000006fd" shape=box style=filled ];
}

subgraph cluster_0x000007d1 { label="function 0x000007d1 \\"CF::<clinit>\\"" fillcolor="#f2f2f2" href="0x000007d1" style=filled
V_0x000007d1 [ label=<000007d1  ?? new    0x0041&lt;java/util/LinkedList&gt;<br align="left"/>000007d4  ?? dup    <br align="left"/>000007d5  ?? invokespecial 0x0043&lt;java/util/LinkedList::&lt;init&gt;&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000007d1" shape=box style=filled ];
V_0x000007d8 [ label=<000007d8  ?? putstatic 0x002a&lt;CF::list&gt;<br align="left"/>000007db  ?? return <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000007d8" shape=box style=filled ];
}

subgraph cluster_0x00001460 { label="function 0x00001460 \\"CF$FastScanner::<init>\\"" fillcolor="#f2f2f2" href="0x00001460" style=filled
V_0x00001460 [ label=<00001460  ?? aload_0 <br align="left"/>00001461  ?? invokespecial 0x0001&lt;java/lang/Object::&lt;init&gt;&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x00001460" shape=box style=filled ];
V_0x00001464 [ label=<00001464  ?? aload_0 <br align="left"/>00001465  ?? sipush 0x0400&lt;1024&gt;<br align="left"/>00001468  ?? newarray 0x08<br align="left"/>0000146a  ?? putfield 0x0007<br align="left"/>0000146d  ?? aload_0 <br align="left"/>0000146e  ?? aload_1 <br align="left"/>0000146f  ?? putfield 0x000d<br align="left"/>00001472  ?? return <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001464" shape=box style=filled ];
}

subgraph cluster_0x000014a5 { label="function 0x000014a5 \\"CF$FastScanner::read\\"" fillcolor="#f2f2f2" href="0x000014a5" style=filled
V_0x000014a5 [ label=<000014a5  ?? aload_0 <br align="left"/>000014a6  ?? getfield 0x0011<br align="left"/>000014a9  ?? iconst_m1 <br align="left"/>000014aa  ?? if_icmpne 0x000b<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000014a5" shape=box style=filled ];
V_0x000014ad [ label=<000014ad  ?? new    0x0015&lt;java/util/InputMismatchException&gt;<br align="left"/>000014b0  ?? dup    <br align="left"/>000014b1  ?? invokespecial 0x0017&lt;java/util/InputMismatchException::&lt;init&gt;&gt;<br align="left"/>> fontname=Courier href="0x000014ad" shape=box ];
V_0x000014b5 [ label=<000014b5  ?? aload_0 <br align="left"/>000014b6  ?? getfield 0x0018<br align="left"/>000014b9  ?? aload_0 <br align="left"/>000014ba  ?? getfield 0x0011<br align="left"/>000014bd  ?? if_icmplt 0x002c<br align="left"/>> fontname=Courier href="0x000014b5" shape=box ];
V_0x000014b4 [ label=<000014b4  ?? athrow <br align="left"/>> fontname=Courier href="0x000014b4" shape=box ];
V_0x000014c0 [ label=<000014c0  ?? aload_0 <br align="left"/>000014c1  ?? iconst_0 <br align="left"/>000014c2  ?? putfield 0x0018<br align="left"/>000014c5  ?? aload_0 <br align="left"/>000014c6  ?? aload_0 <br align="left"/>000014c7  ?? getfield 0x000d<br align="left"/>000014ca  ?? aload_0 <br align="left"/>000014cb  ?? getfield 0x0007<br align="left"/>000014ce  ?? invokevirtual 0x001b&lt;java/io/InputStream::read&gt;<br align="left"/>> fontname=Courier href="0x000014c0" shape=box ];
V_0x000014e9 [ label=<000014e9  ?? aload_0 <br align="left"/>000014ea  ?? getfield 0x0007<br align="left"/>000014ed  ?? aload_0 <br align="left"/>000014ee  ?? dup    <br align="left"/>000014ef  ?? getfield 0x0018<br align="left"/>000014f2  ?? dup_x1 <br align="left"/>000014f3  ?? iconst_1 <br align="left"/>000014f4  ?? iadd   <br align="left"/>000014f5  ?? putfield 0x0018<br align="left"/>000014f8  ?? baload <br align="left"/>000014f9  ?? ireturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000014e9" shape=box style=filled ];
V_0x000014d1 [ label=<000014d1  ?? putfield 0x0011<br align="left"/>000014d4  ?? goto   0x000c<br align="left"/>> fontname=Courier href="0x000014d1" shape=box ];
V_0x000014e0 [ label=<000014e0  ?? aload_0 <br align="left"/>000014e1  ?? getfield 0x0011<br align="left"/>000014e4  ?? ifgt   0x0005<br align="left"/>> fontname=Courier href="0x000014e0" shape=box ];
V_0x000014d7 [ label=<000014d7  ?? astore_1 <br align="left"/>000014d8  ?? new    0x0015&lt;java/util/InputMismatchException&gt;<br align="left"/>000014db  ?? dup    <br align="left"/>000014dc  ?? invokespecial 0x0017&lt;java/util/InputMismatchException::&lt;init&gt;&gt;<br align="left"/>> fontname=Courier href="0x000014d7" shape=box ];
V_0x000014df [ label=<000014df  ?? athrow <br align="left"/>> fontname=Courier href="0x000014df" shape=box ];
V_0x000014e7 [ label=<000014e7  ?? iconst_m1 <br align="left"/>000014e8  ?? ireturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000014e7" shape=box style=filled ];
}

subgraph cluster_0x0000155f { label="function 0x0000155f \\"CF$FastScanner::isSpaceChar\\"" fillcolor="#f2f2f2" href="0x0000155f" style=filled
V_0x0000155f [ label=<0000155f  ?? iload_1 <br align="left"/>00001560  ?? bipush 0x20<br align="left"/>00001562  ?? if_icmpeq 0x001a<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000155f" shape=box style=filled ];
V_0x00001565 [ label=<00001565  ?? iload_1 <br align="left"/>00001566  ?? bipush 0x0a<br align="left"/>00001568  ?? if_icmpeq 0x0014<br align="left"/>> fontname=Courier href="0x00001565" shape=box ];
V_0x0000157c [ label=<0000157c  ?? iconst_1 <br align="left"/>0000157d  ?? goto   0x0004<br align="left"/>> fontname=Courier href="0x0000157c" shape=box ];
V_0x0000156b [ label=<0000156b  ?? iload_1 <br align="left"/>0000156c  ?? bipush 0x0d<br align="left"/>0000156e  ?? if_icmpeq 0x000e<br align="left"/>> fontname=Courier href="0x0000156b" shape=box ];
V_0x00001571 [ label=<00001571  ?? iload_1 <br align="left"/>00001572  ?? bipush 0x09<br align="left"/>00001574  ?? if_icmpeq 0x0008<br align="left"/>> fontname=Courier href="0x00001571" shape=box ];
V_0x00001577 [ label=<00001577  ?? iload_1 <br align="left"/>00001578  ?? iconst_m1 <br align="left"/>00001579  ?? if_icmpne 0x0007<br align="left"/>> fontname=Courier href="0x00001577" shape=box ];
V_0x00001580 [ label=<00001580  ?? iconst_0 <br align="left"/>> fontname=Courier href="0x00001580" shape=box ];
V_0x00001581 [ label=<00001581  ?? ireturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001581" shape=box style=filled ];
}

subgraph cluster_0x000015b4 { label="function 0x000015b4 \\"CF$FastScanner::isEndline\\"" fillcolor="#f2f2f2" href="0x000015b4" style=filled
V_0x000015b4 [ label=<000015b4  ?? iload_1 <br align="left"/>000015b5  ?? bipush 0x0a<br align="left"/>000015b7  ?? if_icmpeq 0x000e<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000015b4" shape=box style=filled ];
V_0x000015ba [ label=<000015ba  ?? iload_1 <br align="left"/>000015bb  ?? bipush 0x0d<br align="left"/>000015bd  ?? if_icmpeq 0x0008<br align="left"/>> fontname=Courier href="0x000015ba" shape=box ];
V_0x000015c5 [ label=<000015c5  ?? iconst_1 <br align="left"/>000015c6  ?? goto   0x0004<br align="left"/>> fontname=Courier href="0x000015c5" shape=box ];
V_0x000015c0 [ label=<000015c0  ?? iload_1 <br align="left"/>000015c1  ?? iconst_m1 <br align="left"/>000015c2  ?? if_icmpne 0x0007<br align="left"/>> fontname=Courier href="0x000015c0" shape=box ];
V_0x000015c9 [ label=<000015c9  ?? iconst_0 <br align="left"/>> fontname=Courier href="0x000015c9" shape=box ];
V_0x000015ca [ label=<000015ca  ?? ireturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000015ca" shape=box style=filled ];
}

subgraph cluster_0x000015fd { label="function 0x000015fd \\"CF$FastScanner::nextInt\\"" fillcolor="#f2f2f2" href="0x000015fd" style=filled
V_0x000015fd [ label=<000015fd  ?? aload_0 <br align="left"/>000015fe  ?? invokevirtual 0x0023&lt;CF$FastScanner::next&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000015fd" shape=box style=filled ];
V_0x00001601 [ label=<00001601  ?? invokestatic 0x0027&lt;java/lang/Integer::parseInt&gt;<br align="left"/>> fontname=Courier href="0x00001601" shape=box ];
V_0x00001604 [ label=<00001604  ?? ireturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001604" shape=box style=filled ];
}

subgraph cluster_0x0000162b { label="function 0x0000162b \\"CF$FastScanner::nextArrayInt\\"" fillcolor="#f2f2f2" href="0x0000162b" style=filled
V_0x0000162b [ label=<0000162b  ?? iload_1 <br align="left"/>0000162c  ?? newarray 0x0a<br align="left"/>0000162e  ?? astore_2 <br align="left"/>0000162f  ?? iconst_0 <br align="left"/>00001630  ?? istore_3 <br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000162b" shape=box style=filled ];
V_0x00001631 [ label=<00001631  ?? iload_3 <br align="left"/>00001632  ?? iload_1 <br align="left"/>00001633  ?? if_icmpge 0x0010<br align="left"/>> fontname=Courier href="0x00001631" shape=box ];
V_0x00001636 [ label=<00001636  ?? aload_2 <br align="left"/>00001637  ?? iload_3 <br align="left"/>00001638  ?? aload_0 <br align="left"/>00001639  ?? invokevirtual 0x002d&lt;CF$FastScanner::nextInt&gt;<br align="left"/>> fontname=Courier href="0x00001636" shape=box ];
V_0x00001643 [ label=<00001643  ?? aload_2 <br align="left"/>00001644  ?? areturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001643" shape=box style=filled ];
V_0x0000163c [ label=<0000163c  ?? iastore <br align="left"/>0000163d  ?? iinc   0x03, 0x01<br align="left"/>00001640  ?? goto   0xfff1&lt;-15&gt;<br align="left"/>> fontname=Courier href="0x0000163c" shape=box ];
}

subgraph cluster_0x0000168d { label="function 0x0000168d \\"CF$FastScanner::nextArrayString\\"" fillcolor="#f2f2f2" href="0x0000168d" style=filled
V_0x0000168d [ label=<0000168d  ?? iload_1 <br align="left"/>0000168e  ?? anewarray 0x0031<br align="left"/>00001691  ?? astore_2 <br align="left"/>00001692  ?? iconst_0 <br align="left"/>00001693  ?? istore_3 <br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000168d" shape=box style=filled ];
V_0x00001694 [ label=<00001694  ?? iload_3 <br align="left"/>00001695  ?? iload_1 <br align="left"/>00001696  ?? if_icmpge 0x0010<br align="left"/>> fontname=Courier href="0x00001694" shape=box ];
V_0x00001699 [ label=<00001699  ?? aload_2 <br align="left"/>0000169a  ?? iload_3 <br align="left"/>0000169b  ?? aload_0 <br align="left"/>0000169c  ?? invokevirtual 0x0023&lt;CF$FastScanner::next&gt;<br align="left"/>> fontname=Courier href="0x00001699" shape=box ];
V_0x000016a6 [ label=<000016a6  ?? aload_2 <br align="left"/>000016a7  ?? areturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000016a6" shape=box style=filled ];
V_0x0000169f [ label=<0000169f  ?? aastore <br align="left"/>000016a0  ?? iinc   0x03, 0x01<br align="left"/>000016a3  ?? goto   0xfff1&lt;-15&gt;<br align="left"/>> fontname=Courier href="0x0000169f" shape=box ];
}

subgraph cluster_0x000016f0 { label="function 0x000016f0 \\"CF$FastScanner::nextLong\\"" fillcolor="#f2f2f2" href="0x000016f0" style=filled
V_0x000016f0 [ label=<000016f0  ?? aload_0 <br align="left"/>000016f1  ?? invokevirtual 0x0023&lt;CF$FastScanner::next&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000016f0" shape=box style=filled ];
V_0x000016f4 [ label=<000016f4  ?? invokestatic 0x0033&lt;java/lang/Long::parseLong&gt;<br align="left"/>> fontname=Courier href="0x000016f4" shape=box ];
V_0x000016f7 [ label=<000016f7  ?? lreturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x000016f7" shape=box style=filled ];
}

subgraph cluster_0x0000171e { label="function 0x0000171e \\"CF$FastScanner::nextDouble\\"" fillcolor="#f2f2f2" href="0x0000171e" style=filled
V_0x0000171e [ label=<0000171e  ?? aload_0 <br align="left"/>0000171f  ?? invokevirtual 0x0023&lt;CF$FastScanner::next&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000171e" shape=box style=filled ];
V_0x00001722 [ label=<00001722  ?? invokestatic 0x0039&lt;java/lang/Double::parseDouble&gt;<br align="left"/>> fontname=Courier href="0x00001722" shape=box ];
V_0x00001725 [ label=<00001725  ?? dreturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001725" shape=box style=filled ];
}

subgraph cluster_0x0000174c { label="function 0x0000174c \\"CF$FastScanner::next\\"" fillcolor="#f2f2f2" href="0x0000174c" style=filled
V_0x0000174c [ label=<0000174c  ?? aload_0 <br align="left"/>0000174d  ?? invokevirtual 0x003f&lt;CF$FastScanner::read&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x0000174c" shape=box style=filled ];
V_0x00001750 [ label=<00001750  ?? istore_1 <br align="left"/>> fontname=Courier href="0x00001750" shape=box ];
V_0x00001751 [ label=<00001751  ?? aload_0 <br align="left"/>00001752  ?? iload_1 <br align="left"/>00001753  ?? invokevirtual 0x0041&lt;CF$FastScanner::isSpaceChar&gt;<br align="left"/>> fontname=Courier href="0x00001751" shape=box ];
V_0x00001756 [ label=<00001756  ?? ifeq   0x000b<br align="left"/>> fontname=Courier href="0x00001756" shape=box ];
V_0x00001759 [ label=<00001759  ?? aload_0 <br align="left"/>0000175a  ?? invokevirtual 0x003f&lt;CF$FastScanner::read&gt;<br align="left"/>> fontname=Courier href="0x00001759" shape=box ];
V_0x00001761 [ label=<00001761  ?? new    0x0045&lt;java/lang/StringBuilder&gt;<br align="left"/>00001764  ?? dup    <br align="left"/>00001765  ?? invokespecial 0x0047&lt;java/lang/StringBuilder::&lt;init&gt;&gt;<br align="left"/>> fontname=Courier href="0x00001761" shape=box ];
V_0x0000175d [ label=<0000175d  ?? istore_1 <br align="left"/>0000175e  ?? goto   0xfff3&lt;-13&gt;<br align="left"/>> fontname=Courier href="0x0000175d" shape=box ];
V_0x00001768 [ label=<00001768  ?? astore_2 <br align="left"/>> fontname=Courier href="0x00001768" shape=box ];
V_0x00001769 [ label=<00001769  ?? aload_2 <br align="left"/>0000176a  ?? iload_1 <br align="left"/>0000176b  ?? invokevirtual 0x0048&lt;java/lang/StringBuilder::appendCodePoint&gt;<br align="left"/>> fontname=Courier href="0x00001769" shape=box ];
V_0x0000176e [ label=<0000176e  ?? pop    <br align="left"/>0000176f  ?? aload_0 <br align="left"/>00001770  ?? invokevirtual 0x003f&lt;CF$FastScanner::read&gt;<br align="left"/>> fontname=Courier href="0x0000176e" shape=box ];
V_0x00001773 [ label=<00001773  ?? istore_1 <br align="left"/>00001774  ?? aload_0 <br align="left"/>00001775  ?? iload_1 <br align="left"/>00001776  ?? invokevirtual 0x0041&lt;CF$FastScanner::isSpaceChar&gt;<br align="left"/>> fontname=Courier href="0x00001773" shape=box ];
V_0x00001779 [ label=<00001779  ?? ifeq   0xfff0&lt;-16&gt;<br align="left"/>> fontname=Courier href="0x00001779" shape=box ];
V_0x0000177c [ label=<0000177c  ?? aload_2 <br align="left"/>0000177d  ?? invokevirtual 0x004c&lt;java/lang/StringBuilder::toString&gt;<br align="left"/>> fontname=Courier href="0x0000177c" shape=box ];
V_0x00001780 [ label=<00001780  ?? areturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x00001780" shape=box style=filled ];
}

subgraph cluster_0x000017d6 { label="function 0x000017d6 \\"CF$FastScanner::nextLine\\"" fillcolor="#f2f2f2" href="0x000017d6" style=filled
V_0x000017d6 [ label=<000017d6  ?? aload_0 <br align="left"/>000017d7  ?? invokevirtual 0x003f&lt;CF$FastScanner::read&gt;<br align="left"/>> fillcolor="#cdfecc" fontname=Courier href="0x000017d6" shape=box style=filled ];
V_0x000017da [ label=<000017da  ?? istore_1 <br align="left"/>> fontname=Courier href="0x000017da" shape=box ];
V_0x000017db [ label=<000017db  ?? aload_0 <br align="left"/>000017dc  ?? iload_1 <br align="left"/>000017dd  ?? invokevirtual 0x004f&lt;CF$FastScanner::isEndline&gt;<br align="left"/>> fontname=Courier href="0x000017db" shape=box ];
V_0x000017e0 [ label=<000017e0  ?? ifeq   0x000b<br align="left"/>> fontname=Courier href="0x000017e0" shape=box ];
V_0x000017e3 [ label=<000017e3  ?? aload_0 <br align="left"/>000017e4  ?? invokevirtual 0x003f&lt;CF$FastScanner::read&gt;<br align="left"/>> fontname=Courier href="0x000017e3" shape=box ];
V_0x000017eb [ label=<000017eb  ?? new    0x0045&lt;java/lang/StringBuilder&gt;<br align="left"/>000017ee  ?? dup    <br align="left"/>000017ef  ?? invokespecial 0x0047&lt;java/lang/StringBuilder::&lt;init&gt;&gt;<br align="left"/>> fontname=Courier href="0x000017eb" shape=box ];
V_0x000017e7 [ label=<000017e7  ?? istore_1 <br align="left"/>000017e8  ?? goto   0xfff3&lt;-13&gt;<br align="left"/>> fontname=Courier href="0x000017e7" shape=box ];
V_0x000017f2 [ label=<000017f2  ?? astore_2 <br align="left"/>> fontname=Courier href="0x000017f2" shape=box ];
V_0x000017f3 [ label=<000017f3  ?? aload_2 <br align="left"/>000017f4  ?? iload_1 <br align="left"/>000017f5  ?? invokevirtual 0x0048&lt;java/lang/StringBuilder::appendCodePoint&gt;<br align="left"/>> fontname=Courier href="0x000017f3" shape=box ];
V_0x000017f8 [ label=<000017f8  ?? pop    <br align="left"/>000017f9  ?? aload_0 <br align="left"/>000017fa  ?? invokevirtual 0x003f&lt;CF$FastScanner::read&gt;<br align="left"/>> fontname=Courier href="0x000017f8" shape=box ];
V_0x000017fd [ label=<000017fd  ?? istore_1 <br align="left"/>000017fe  ?? aload_0 <br align="left"/>000017ff  ?? iload_1 <br align="left"/>00001800  ?? invokevirtual 0x004f&lt;CF$FastScanner::isEndline&gt;<br align="left"/>> fontname=Courier href="0x000017fd" shape=box ];
V_0x00001803 [ label=<00001803  ?? ifeq   0xfff0&lt;-16&gt;<br align="left"/>> fontname=Courier href="0x00001803" shape=box ];
V_0x00001806 [ label=<00001806  ?? aload_2 <br align="left"/>00001807  ?? invokevirtual 0x004c&lt;java/lang/StringBuilder::toString&gt;<br align="left"/>> fontname=Courier href="0x00001806" shape=box ];
V_0x0000180a [ label=<0000180a  ?? areturn <br align="left"/>> fillcolor="#cdccfe" fontname=Courier href="0x0000180a" shape=box style=filled ];
}

subgraph cluster_0xffffffffffff57ff { label="function 0xffffffffffff57ff \\"java/lang/StringBuilder::toString\\"" fillcolor="#f2f2f2" href="0xffffffffffff57ff" style=filled
V_0xffffffffffff57ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffff57ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffff63ff { label="function 0xffffffffffff63ff \\"java/lang/StringBuilder::appendCodePoint\\"" fillcolor="#f2f2f2" href="0xffffffffffff63ff" style=filled
V_0xffffffffffff63ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffff63ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffff67ff { label="function 0xffffffffffff67ff \\"java/lang/StringBuilder::<init>\\"" fillcolor="#f2f2f2" href="0xffffffffffff67ff" style=filled
V_0xffffffffffff67ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffff67ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffff97ff { label="function 0xffffffffffff97ff \\"java/lang/Double::parseDouble\\"" fillcolor="#f2f2f2" href="0xffffffffffff97ff" style=filled
V_0xffffffffffff97ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffff97ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffff9fff { label="function 0xffffffffffff9fff \\"java/lang/Long::parseLong\\"" fillcolor="#f2f2f2" href="0xffffffffffff9fff" style=filled
V_0xffffffffffff9fff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffff9fff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffafff { label="function 0xffffffffffffafff \\"java/lang/Integer::parseInt\\"" fillcolor="#f2f2f2" href="0xffffffffffffafff" style=filled
V_0xffffffffffffafff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffafff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffb7ff { label="function 0xffffffffffffb7ff \\"java/util/InputMismatchException::<init>\\"" fillcolor="#f2f2f2" href="0xffffffffffffb7ff" style=filled
V_0xffffffffffffb7ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffb7ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffbbff { label="function 0xffffffffffffbbff \\"java/io/InputStream::read\\"" fillcolor="#f2f2f2" href="0xffffffffffffbbff" style=filled
V_0xffffffffffffbbff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffbbff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffc3ff { label="function 0xffffffffffffc3ff \\"java/lang/Object::<init>\\"" fillcolor="#f2f2f2" href="0xffffffffffffc3ff" style=filled
V_0xffffffffffffc3ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffc3ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffc7ff { label="function 0xffffffffffffc7ff \\"java/util/LinkedList::<init>\\"" fillcolor="#f2f2f2" href="0xffffffffffffc7ff" style=filled
V_0xffffffffffffc7ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffc7ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffd3ff { label="function 0xffffffffffffd3ff \\"java/util/List::add\\"" fillcolor="#f2f2f2" href="0xffffffffffffd3ff" style=filled
V_0xffffffffffffd3ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffd3ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffd7ff { label="function 0xffffffffffffd7ff \\"java/io/PrintStream::println\\"" fillcolor="#f2f2f2" href="0xffffffffffffd7ff" style=filled
V_0xffffffffffffd7ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffd7ff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffdbff { label="function 0xffffffffffffdbff \\"java/lang/Math::min\\"" fillcolor="#f2f2f2" href="0xffffffffffffdbff" style=filled
V_0xffffffffffffdbff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffdbff" shape=box style=filled ];
}

subgraph cluster_0xffffffffffffe3ff { label="function 0xffffffffffffe3ff \\"java/lang/Math::abs\\"" fillcolor="#f2f2f2" href="0xffffffffffffe3ff" style=filled
V_0xffffffffffffe3ff [ label=<(no insns)> fillcolor="#cdfecc" fontname=Courier href="0xffffffffffffe3ff" shape=box style=filled ];
}
indeterminate [ label="indeterminate" fillcolor="#ff9999" shape=box style=filled ];
nonexisting [ label="nonexisting" fillcolor="#ff9999" shape=box style=filled ];
V_0xffffffffffffbbff -> nonexisting [ label="other"  ];
V_0xffffffffffffd7ff -> nonexisting [ label="other"  ];
V_0xffffffffffff97ff -> nonexisting [ label="other"  ];
V_0xffffffffffffafff -> nonexisting [ label="other"  ];
V_0xffffffffffff9fff -> nonexisting [ label="other"  ];
V_0xffffffffffffe3ff -> nonexisting [ label="other"  ];
V_0xffffffffffffdbff -> nonexisting [ label="other"  ];
V_0xffffffffffffc3ff -> nonexisting [ label="other"  ];
V_0xffffffffffff67ff -> nonexisting [ label="other"  ];
V_0xffffffffffff63ff -> nonexisting [ label="other"  ];
V_0xffffffffffff57ff -> nonexisting [ label="other"  ];
V_0xffffffffffffb7ff -> nonexisting [ label="other"  ];
V_0xffffffffffffc7ff -> nonexisting [ label="other"  ];
V_0xffffffffffffd3ff -> nonexisting [ label="other"  ];
V_0x00001460 -> V_0x00001464 [ label="cret" style=dotted ];
V_0x00001460 -> V_0xffffffffffffc3ff [ label="call" color="#05ff00" ];
V_0x000014ad -> V_0x000014b4 [ label="cret" style=dotted ];
V_0x000014a5 -> V_0x000014ad [ label="" style=dotted ];
V_0x000014a5 -> V_0x000014b5 [ label=""  ];
V_0x000014ad -> V_0xffffffffffffb7ff [ label="call" color="#05ff00" ];
V_0x000014b4 -> V_0x000014b5 [ label="" style=dotted ];
V_0x000014e0 -> V_0x000014e9 [ label=""  ];
V_0x000014b5 -> V_0x000014c0 [ label="" style=dotted ];
V_0x000014b5 -> V_0x000014e9 [ label=""  ];
V_0x000014c0 -> V_0xffffffffffffbbff [ label="call" color="#05ff00" ];
V_0x000014c0 -> V_0x000014d1 [ label="cret" style=dotted ];
V_0x000014d1 -> V_0x000014e0 [ label=""  ];
V_0x000014d7 -> V_0x000014df [ label="cret" style=dotted ];
V_0x000014d7 -> V_0xffffffffffffb7ff [ label="call" color="#05ff00" ];
V_0x000014df -> V_0x000014e0 [ label="" style=dotted ];
V_0x000014e0 -> V_0x000014e7 [ label="" style=dotted ];
V_0x00001580 -> V_0x00001581 [ label="" style=dotted ];
V_0x0000157c -> V_0x00001581 [ label=""  ];
V_0x0000155f -> V_0x00001565 [ label="" style=dotted ];
V_0x0000155f -> V_0x0000157c [ label=""  ];
V_0x00001565 -> V_0x0000156b [ label="" style=dotted ];
V_0x00001565 -> V_0x0000157c [ label=""  ];
V_0x0000156b -> V_0x00001571 [ label="" style=dotted ];
V_0x0000156b -> V_0x0000157c [ label=""  ];
V_0x00001571 -> V_0x00001577 [ label="" style=dotted ];
V_0x00001571 -> V_0x0000157c [ label=""  ];
V_0x00001577 -> V_0x0000157c [ label="" style=dotted ];
V_0x00001577 -> V_0x00001580 [ label=""  ];
V_0x000015c9 -> V_0x000015ca [ label="" style=dotted ];
V_0x000015c5 -> V_0x000015ca [ label=""  ];
V_0x000015b4 -> V_0x000015ba [ label="" style=dotted ];
V_0x000015b4 -> V_0x000015c5 [ label=""  ];
V_0x000015ba -> V_0x000015c0 [ label="" style=dotted ];
V_0x000015ba -> V_0x000015c5 [ label=""  ];
V_0x000015c0 -> V_0x000015c5 [ label="" style=dotted ];
V_0x000015c0 -> V_0x000015c9 [ label=""  ];
V_0x00001601 -> V_0x00001604 [ label="cret" style=dotted ];
V_0x000015fd -> indeterminate [ label="call" color="#05ff00" ];
V_0x000015fd -> V_0x00001601 [ label="cret" style=dotted ];
V_0x00001601 -> V_0xffffffffffffafff [ label="call" color="#05ff00" ];
V_0x0000163c -> V_0x00001631 [ label=""  ];
V_0x0000162b -> V_0x00001631 [ label="" style=dotted ];
V_0x00001636 -> V_0x0000163c [ label="cret" style=dotted ];
V_0x00001631 -> V_0x00001636 [ label="" style=dotted ];
V_0x00001631 -> V_0x00001643 [ label=""  ];
V_0x00001636 -> V_0x000015fd [ label="call" color="#05ff00" ];
V_0x0000169f -> V_0x00001694 [ label=""  ];
V_0x0000168d -> V_0x00001694 [ label="" style=dotted ];
V_0x00001699 -> V_0x0000169f [ label="cret" style=dotted ];
V_0x00001694 -> V_0x00001699 [ label="" style=dotted ];
V_0x00001694 -> V_0x000016a6 [ label=""  ];
V_0x00001699 -> indeterminate [ label="call" color="#05ff00" ];
V_0x000016f4 -> V_0x000016f7 [ label="cret" style=dotted ];
V_0x000016f0 -> indeterminate [ label="call" color="#05ff00" ];
V_0x000016f0 -> V_0x000016f4 [ label="cret" style=dotted ];
V_0x000016f4 -> V_0xffffffffffff9fff [ label="call" color="#05ff00" ];
V_0x00001722 -> V_0x00001725 [ label="cret" style=dotted ];
V_0x0000171e -> indeterminate [ label="call" color="#05ff00" ];
V_0x0000171e -> V_0x00001722 [ label="cret" style=dotted ];
V_0x00001722 -> V_0xffffffffffff97ff [ label="call" color="#05ff00" ];
V_0x0000177c -> V_0x00001780 [ label="cret" style=dotted ];
V_0x0000174c -> V_0x000014a5 [ label="call" color="#05ff00" ];
V_0x0000174c -> V_0x00001750 [ label="cret" style=dotted ];
V_0x00001750 -> V_0x00001751 [ label="" style=dotted ];
V_0x00001751 -> V_0x0000155f [ label="call" color="#05ff00" ];
V_0x00001751 -> V_0x00001756 [ label="cret" style=dotted ];
V_0x00001759 -> V_0x0000175d [ label="cret" style=dotted ];
V_0x00001756 -> V_0x00001759 [ label="" style=dotted ];
V_0x00001756 -> V_0x00001761 [ label=""  ];
V_0x00001759 -> V_0x000014a5 [ label="call" color="#05ff00" ];
V_0x0000175d -> V_0x00001751 [ label=""  ];
V_0x00001761 -> V_0xffffffffffff67ff [ label="call" color="#05ff00" ];
V_0x00001761 -> V_0x00001768 [ label="cret" style=dotted ];
V_0x00001768 -> V_0x00001769 [ label="" style=dotted ];
V_0x00001769 -> V_0xffffffffffff63ff [ label="call" color="#05ff00" ];
V_0x00001769 -> V_0x0000176e [ label="cret" style=dotted ];
V_0x0000176e -> V_0x000014a5 [ label="call" color="#05ff00" ];
V_0x0000176e -> V_0x00001773 [ label="cret" style=dotted ];
V_0x00001773 -> V_0x0000155f [ label="call" color="#05ff00" ];
V_0x00001773 -> V_0x00001779 [ label="cret" style=dotted ];
V_0x00001779 -> V_0x00001769 [ label=""  ];
V_0x00001779 -> V_0x0000177c [ label="" style=dotted ];
V_0x0000177c -> V_0xffffffffffff57ff [ label="call" color="#05ff00" ];
V_0x00001806 -> V_0x0000180a [ label="cret" style=dotted ];
V_0x000017d6 -> V_0x000014a5 [ label="call" color="#05ff00" ];
V_0x000017d6 -> V_0x000017da [ label="cret" style=dotted ];
V_0x000017da -> V_0x000017db [ label="" style=dotted ];
V_0x000017db -> V_0x000015b4 [ label="call" color="#05ff00" ];
V_0x000017db -> V_0x000017e0 [ label="cret" style=dotted ];
V_0x000017e3 -> V_0x000017e7 [ label="cret" style=dotted ];
V_0x000017e0 -> V_0x000017e3 [ label="" style=dotted ];
V_0x000017e0 -> V_0x000017eb [ label=""  ];
V_0x000017e3 -> V_0x000014a5 [ label="call" color="#05ff00" ];
V_0x000017e7 -> V_0x000017db [ label=""  ];
V_0x000017eb -> V_0xffffffffffff67ff [ label="call" color="#05ff00" ];
V_0x000017eb -> V_0x000017f2 [ label="cret" style=dotted ];
V_0x000017f2 -> V_0x000017f3 [ label="" style=dotted ];
V_0x000017f3 -> V_0xffffffffffff63ff [ label="call" color="#05ff00" ];
V_0x000017f3 -> V_0x000017f8 [ label="cret" style=dotted ];
V_0x000017f8 -> V_0x000014a5 [ label="call" color="#05ff00" ];
V_0x000017f8 -> V_0x000017fd [ label="cret" style=dotted ];
V_0x000017fd -> V_0x000015b4 [ label="call" color="#05ff00" ];
V_0x000017fd -> V_0x00001803 [ label="cret" style=dotted ];
V_0x00001803 -> V_0x000017f3 [ label=""  ];
V_0x00001803 -> V_0x00001806 [ label="" style=dotted ];
V_0x00001806 -> V_0xffffffffffff57ff [ label="call" color="#05ff00" ];
V_0x00000386 -> V_0x0000038a [ label="cret" style=dotted ];
V_0x00000386 -> V_0xffffffffffffc3ff [ label="call" color="#05ff00" ];
V_0x000003f0 -> V_0x00000406 [ label=""  ];
V_0x000003b1 -> V_0x00001460 [ label="call" color="#05ff00" ];
V_0x000003b1 -> V_0x000003bb [ label="cret" style=dotted ];
V_0x000003bb -> V_0x000015fd [ label="call" color="#05ff00" ];
V_0x000003bb -> V_0x000003c0 [ label="cret" style=dotted ];
V_0x000003c0 -> V_0x000015fd [ label="call" color="#05ff00" ];
V_0x000003c0 -> V_0x000003c5 [ label="cret" style=dotted ];
V_0x000003c5 -> V_0x000015fd [ label="call" color="#05ff00" ];
V_0x000003c5 -> V_0x000003ca [ label="cret" style=dotted ];
V_0x000003ca -> V_0x000015fd [ label="call" color="#05ff00" ];
V_0x000003ca -> V_0x000003d0 [ label="cret" style=dotted ];
V_0x00000424 -> V_0x0000043a [ label=""  ];
V_0x000003d0 -> V_0x000003db [ label="" style=dotted ];
V_0x000003d0 -> V_0x00000400 [ label=""  ];
V_0x000003db -> V_0xffffffffffffe3ff [ label="call" color="#05ff00" ];
V_0x000003db -> V_0x000003e4 [ label="cret" style=dotted ];
V_0x000003e4 -> V_0x00000406 [ label=""  ];
V_0x000003e4 -> V_0x000003f0 [ label="" style=dotted ];
V_0x00000400 -> V_0x00000406 [ label="" style=dotted ];
V_0x0000044f -> V_0x00000459 [ label="cret" style=dotted ];
V_0x00000406 -> V_0x0000040f [ label="" style=dotted ];
V_0x00000406 -> V_0x00000434 [ label=""  ];
V_0x0000040f -> V_0xffffffffffffe3ff [ label="call" color="#05ff00" ];
V_0x0000040f -> V_0x00000418 [ label="cret" style=dotted ];
V_0x00000418 -> V_0x0000043a [ label=""  ];
V_0x00000418 -> V_0x00000424 [ label="" style=dotted ];
V_0x00000434 -> V_0x0000043a [ label="" style=dotted ];
V_0x00000459 -> V_0x0000045c [ label="cret" style=dotted ];
V_0x0000043a -> V_0x00000445 [ label="" style=dotted ];
V_0x0000043a -> V_0x0000044f [ label=""  ];
V_0x00000445 -> V_0xffffffffffffd7ff [ label="call" color="#05ff00" ];
V_0x00000445 -> V_0x0000044c [ label="cret" style=dotted ];
V_0x0000044c -> V_0x0000045c [ label=""  ];
V_0x0000044f -> V_0xffffffffffffdbff [ label="call" color="#05ff00" ];
V_0x00000459 -> V_0xffffffffffffd7ff [ label="call" color="#05ff00" ];
V_0x00000459 -> V_0x0000045c [ label="" style=dotted ];
V_0x00000549 -> V_0x00000560 [ label="cret" style=dotted ];
V_0x00000560 -> V_0x00000544 [ label=""  ];
V_0x0000050f -> V_0x00000519 [ label="" style=dotted ];
V_0x0000050f -> V_0x00000542 [ label=""  ];
V_0x00000519 -> V_0x00000520 [ label="" style=dotted ];
V_0x00000520 -> V_0x00000535 [ label=""  ];
V_0x00000520 -> V_0x00000527 [ label="" style=dotted ];
V_0x00000527 -> V_0x00000520 [ label=""  ];
V_0x00000535 -> V_0xffffffffffffd3ff [ label="call" color="#05ff00" ];
V_0x00000535 -> V_0x0000053e [ label="cret" style=dotted ];
V_0x0000053e -> V_0x00000576 [ label=""  ];
V_0x00000542 -> V_0x00000544 [ label="" style=dotted ];
V_0x00000544 -> V_0x00000549 [ label="" style=dotted ];
V_0x00000544 -> V_0x00000576 [ label=""  ];
V_0x00000549 -> V_0x0000050f [ label="call" color="#05ff00" ];
V_0x000005fe -> V_0x00000604 [ label="cret" style=dotted ];
V_0x000005fe -> V_0x000005fe [ label="call" color="#05ff00" ];
V_0x000006e3 -> V_0x000006dd [ label=""  ];
V_0x0000062b -> V_0x00000637 [ label="" style=dotted ];
V_0x00000637 -> V_0x00000652 [ label=""  ];
V_0x00000637 -> V_0x0000063d [ label="" style=dotted ];
V_0x0000063d -> V_0x00000637 [ label=""  ];
V_0x00000652 -> V_0x00000655 [ label="" style=dotted ];
V_0x00000655 -> V_0x0000066f [ label=""  ];
V_0x00000655 -> V_0x0000065c [ label="" style=dotted ];
V_0x0000065c -> V_0x00000655 [ label=""  ];
V_0x0000066f -> V_0x00000672 [ label="" style=dotted ];
V_0x00000672 -> V_0x00000692 [ label=""  ];
V_0x00000672 -> V_0x00000678 [ label="" style=dotted ];
V_0x00000678 -> V_0x00000672 [ label=""  ];
V_0x00000692 -> V_0x000006a2 [ label="" style=dotted ];
V_0x000006a2 -> V_0x000006bd [ label=""  ];
V_0x000006a2 -> V_0x000006a8 [ label="" style=dotted ];
V_0x000006a8 -> V_0x000006a2 [ label=""  ];
V_0x000006bd -> V_0x000006c0 [ label="" style=dotted ];
V_0x000006c0 -> V_0x000006da [ label=""  ];
V_0x000006c0 -> V_0x000006c7 [ label="" style=dotted ];
V_0x000006c7 -> V_0x000006c0 [ label=""  ];
V_0x000006da -> V_0x000006dd [ label="" style=dotted ];
V_0x000006dd -> V_0x000006fd [ label=""  ];
V_0x000006dd -> V_0x000006e3 [ label="" style=dotted ];
V_0x000007d1 -> V_0x000007d8 [ label="cret" style=dotted ];
V_0x000007d1 -> V_0xffffffffffffc7ff [ label="call" color="#05ff00" ];
}
"""
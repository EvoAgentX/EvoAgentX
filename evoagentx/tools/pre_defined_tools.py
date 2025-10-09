
from evoagentx.tools.browser_browseruse_auto import BrowserUseAutoToolkit
# from evoagentx.tools.collection_image import ImageCollectionToolkit
from evoagentx.tools.collection_image_analysis import ImageAnalysisCollectionToolkit 
from evoagentx.tools.collection_image_generation import ImageGenerationCollectionToolkit
from evoagentx.tools.collection_image_edit import ImageEditingCollectionToolkit
from evoagentx.tools.collection_search import SearchCollectionToolkit
from evoagentx.tools.crawler_crawl4ai import Crawl4AICrawlToolkit
from evoagentx.tools.crawler_request import RequestCrawlToolkit
from evoagentx.tools.request import RequestToolkit
from evoagentx.tools.rss_feed import RSSToolkit
from evoagentx.tools.storage_file import StorageToolkit
from evoagentx.tools.request_arxiv import ArxivToolkit
from evoagentx.tools.cmd_toolkit import CMDToolkit



default_tools = [
    ## Avaliable Tools
    SearchCollectionToolkit(),
    Crawl4AICrawlToolkit(),
    RequestCrawlToolkit(),
    RSSToolkit(),
    RequestToolkit(),
    BrowserUseAutoToolkit(),
    
    ## Storage Toolkits
    StorageToolkit(),
    ArxivToolkit(),
    CMDToolkit(),
    # ImageCollectionToolkit(),
    ImageAnalysisCollectionToolkit(),
    ImageGenerationCollectionToolkit(),
    ImageEditingCollectionToolkit(),
]
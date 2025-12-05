# d:\pandasai\arsipy\utils\web_search.py
import requests
from bs4 import BeautifulSoup
from typing import List, Dict, Optional
import logging
from urllib.parse import quote_plus, urlparse, urljoin
import traceback
import time
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class ArchivalWebSearch:
    """Specialized web search focused on specific archival and standards resources."""

    TRUSTED_DOMAINS = [
        'ica.org',          # International Council on Archives (Primarily English)
        'archivists.org',   # Society of American Archivists (Primarily English)
        'anri.go.id',       # Arsip Nasional RI (Primarily Indonesian)
        'iso.org',          # ISO Standards (Primarily English)
    ]
    ENGLISH_QUERY_SOURCES = {'ICA', 'SAA', 'ISO'}

    # --- MODIFIED METHOD ---
    def _translate_query_if_needed(self, query: str) -> Optional[str]:
        """
        Assumes the query is Indonesian and attempts to translate it to English using an LLM.
        Bypasses language detection.
        """
        logger.info("Assuming query is Indonesian, attempting translation to English...")
        try:
            # Directly attempt translation using the LLM
            translate_llm = ChatGroq(
                groq_api_key=os.getenv('GROQ_API_KEY'),
                model_name="llama3-8b-8192", # Using a fast model for translation
                temperature=0.1
            )
            prompt_template = ChatPromptTemplate.from_template(
                "Translate the following Indonesian text to English. Output only the translated text, nothing else:\n\n{text}"
            )
            chain = prompt_template | translate_llm
            response = chain.invoke({"text": query})

            if response and hasattr(response, 'content'):
                translated_query = response.content.strip()
                # Important Check: Only return if translation is successful AND different from original
                if translated_query and translated_query.lower() != query.lower():
                    logger.info(f"Successfully translated query to: '{translated_query}'")
                    return translated_query
                else:
                    # Log if translation was empty or same as original (e.g., if input was already English)
                    logger.warning(f"Translation resulted in empty or identical query ('{translated_query}'). Using original query for English sources.")
                    return None # Indicate translation wasn't useful/successful
            else:
                logger.error("LLM translation response was empty or invalid.")
                return None # Indicate translation failed
        except Exception as llm_err:
            # Log any error during the LLM call
            logger.error(f"LLM translation failed: {llm_err}")
            logger.debug(traceback.format_exc()) # Add traceback for debugging if needed
            return None # Indicate translation failed
    # --- END MODIFIED METHOD ---


    def search(self, query: str, max_results_per_source: int = 2, total_max_results: int = 5) -> List[Dict]:
        """Search specified archival/standards websites and return relevant content, with auto-translation."""
        results = []
        original_encoded_query = quote_plus(query)

        # --- Translation attempt happens here, assuming Indonesian ---
        translated_english_query = self._translate_query_if_needed(query)
        encoded_english_query = quote_plus(translated_english_query) if translated_english_query else None
        # ---

        search_endpoint_templates = {
            'ICA': "https://www.ica.org/en/search-results?search_api_fulltext={query_param}",
            'SAA': "https://www2.archivists.org/search/saasearch_simple/{query_param}",
            'ANRI': "https://anri.go.id/search?q={query_param}",
            'ISO': "https://www.iso.org/search.html?query={query_param}"
        }

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'Accept-Language': 'en-US,en;q=0.9,id;q=0.8',
            'Referer': 'https://www.google.com/',
            'DNT': '1',
            'Upgrade-Insecure-Requests': '1'
        }

        for source_name, url_template in search_endpoint_templates.items():
            if len(results) >= total_max_results:
                break

            current_encoded_query = original_encoded_query # Default to original query
            query_log_info = f"(Original: '{query}')"

            # Use translated query ONLY for designated English sources AND if translation was successful/different
            if source_name in self.ENGLISH_QUERY_SOURCES and encoded_english_query:
                current_encoded_query = encoded_english_query # Use the translated query
                query_log_info = f"(Translated: '{translated_english_query}')"
            # No need for elif/else, original is default

            url = url_template.format(query_param=current_encoded_query)
            specific_headers = headers.copy()
            if source_name == 'ANRI':
                specific_headers['Referer'] = 'https://anri.go.id/'

            logger.info(f"Searching {source_name} {query_log_info} with URL: {url}")
            try:
                # ... (rest of the search and parsing logic remains the same) ...
                # time.sleep(0.5) # Optional delay
                response = requests.get(url, headers=specific_headers, timeout=25)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                count_for_this_source = 0

                # --- Parsing Logic ---
                potential_containers = soup.select(
                    # Add/refine selectors based on inspection of target sites
                    'article.search-result, div.result-item, li.entry, div.result, '
                    'div.search__result, li.search-results__item, ' # ICA specific
                    'div.node-search-result, ' # SAA specific (Keep this one)
                    'div.gsc-webResult, div.gsc-result, ' # Google Custom Search (ANRI)
                    'div.search-results-item' # ISO specific
                )
                if not potential_containers:
                     potential_containers = soup.find_all(['article', 'div', 'li'], limit=15)
                     logger.debug(f"Using broader fallback selectors for {source_name}")

                logger.debug(f"Found {len(potential_containers)} potential result containers on {source_name}")

                for item in potential_containers:
                    if count_for_this_source >= max_results_per_source or len(results) >= total_max_results:
                        break

                    title = "No Title Found"
                    content = "No Snippet Found"
                    link_url = ''
                    absolute_link_url = ''
                    link_tag = None # Initialize link_tag

                    try:
                        # --- Find Title (More specific selectors might be needed per source) ---
                        # Prioritize specific title classes/tags if known
                        title_tag = item.select_one('h2.title a, h3.title a, a.gs-title, a.search-result__title, a.result__title') or \
                                    item.find(['h2', 'h3']) # Fallback to generic h2/h3

                        if title_tag:
                            title = title_tag.get_text(strip=True)
                            # --- *** MODIFICATION START *** ---
                            # --- PRIORITIZE LINK FROM TITLE TAG ITSELF or its direct parent if title is not 'a' ---
                            if title_tag.name == 'a' and title_tag.has_attr('href'):
                                link_tag = title_tag # Title itself is the link
                            else:
                                # Maybe the title text is inside a link within the h2/h3?
                                inner_link = title_tag.find('a', href=True)
                                if inner_link:
                                     link_tag = inner_link
                                else:
                                     # Check if the immediate parent of the h2/h3 is a link
                                     parent_link = title_tag.find_parent('a', href=True)
                                     if parent_link:
                                         link_tag = parent_link
                            # --- *** MODIFICATION END *** ---

                        # --- Find Content Snippet (Keep existing logic) ---
                        content_tag = item.select_one('p.summary, p.description, div.content, div.snippet, div.gs-snippet, div.search-result__snippet, div.search-result__body') or \
                                      item.find('p')
                        if content_tag:
                            content = content_tag.get_text(strip=True)
                            content = ' '.join(content.split())

                        # --- Find Link (Fallback if not found via title logic) ---
                        if not link_tag:
                            # Fallback: Find the first plausible link within the item, avoiding small utility links if possible
                            logger.debug(f"Link not found via title for '{title[:30]}...'. Falling back to first link in item.")
                            all_links = item.find_all('a', href=True)
                            for potential_link in all_links:
                                href = potential_link['href']
                                link_text = potential_link.get_text(strip=True).lower()
                                # Try to avoid generic links like "read more", "details", or empty links
                                if href and not href.startswith(('#', 'javascript:')) and len(link_text) > 2 and link_text not in ['read more', 'details', 'continue reading', 'view']:
                                    link_tag = potential_link
                                    break # Take the first plausible one
                            if not link_tag and all_links: # If still no good link, take the very first one as ultimate fallback
                                 link_tag = all_links[0]


                        # --- Process Link Tag (If found) ---
                        if link_tag and link_tag.has_attr('href'):
                            link_url = link_tag['href']

                            # --- Validate and Make URL Absolute ---
                            if link_url and not link_url.startswith(('javascript:', '#', 'mailto:')):
                                parsed_original_url = urlparse(url)
                                base_url = f"{parsed_original_url.scheme}://{parsed_original_url.netloc}"
                                absolute_link_url = urljoin(base_url, link_url)
                                if not absolute_link_url.startswith('http'):
                                     logger.warning(f"Generated invalid absolute URL '{absolute_link_url}' from base '{base_url}' and link '{link_url}' on {source_name}")
                                     absolute_link_url = ''
                                else:
                                     # Check if the link points back to the search results page itself (simple check)
                                     if urlparse(absolute_link_url).path == parsed_original_url.path and urlparse(absolute_link_url).query:
                                         logger.warning(f"Link '{absolute_link_url}' appears to be a search results link, potentially incorrect. Keeping for now.")
                                         # Could potentially invalidate here if this is consistently wrong: absolute_link_url = ''
                        else:
                             logger.debug(f"No valid link tag found for item with title '{title[:30]}...'")


                        # --- Add Result if Valid ---
                        # Ensure we have an absolute URL before adding
                        if absolute_link_url and (title != "No Title Found" or content != "No Snippet Found"):
                            if len(content) < 10 and content == "No Snippet Found":
                                logger.debug(f"Skipping result with potentially empty snippet from {source_name}: {title[:50]}...")
                                continue

                            results.append({
                                'source': source_name,
                                'title': title,
                                'content': content,
                                'url': absolute_link_url,
                                'reliability': 0.9
                            })
                            count_for_this_source += 1
                            logger.debug(f"Added result from {source_name}: {title[:50]}... URL: {absolute_link_url}")
                        else:
                             logger.debug(f"Skipping item from {source_name} due to missing link or content. Title: {title[:50]}...")

                    except Exception as parse_item_err:
                        logger.warning(f"Error parsing a specific item from {source_name}: {parse_item_err}. Skipping item.")
                        logger.debug(traceback.format_exc()) # Log traceback for parsing errors

            # --- (Keep existing exception handling for requests/HTTP errors) ---
            except requests.exceptions.HTTPError as http_err:
                logger.error(f"HTTP error for {source_name} ({url}): {http_err}")
                if http_err.response.status_code == 403:
                    logger.warning(f"ANRI search likely blocked (403 Forbidden).")
            except requests.exceptions.RequestException as req_err:
                logger.error(f"Web search request error for {source_name} ({url}): {req_err}")
            except Exception as e:
                logger.error(f"General error processing source {source_name}: {e}")
                logger.error(traceback.format_exc())

        logger.info(f"Web search completed. Total results found: {len(results)}")
        return results[:total_max_results]